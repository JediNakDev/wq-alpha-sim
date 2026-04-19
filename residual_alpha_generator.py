"""
Residual Alpha Generator (Sector/Size-Neutralized)
===================================================
Hypothesis: Isolating the idiosyncratic component of a raw signal — by
regressing it against its sector/size peer-group mean — produces a cleaner,
neutralized alpha.

Template:
    data = winsorize(ts_backfill(<data>, 63), std=4.0);
    data_gpm = group_mean(data, log(ts_mean(cap, 21)), sector);
    ts_regression(data, data_gpm, 252, rettype=0)

Categories: fundamental, analyst, sentiment, price.

Usage:
    python3 residual_alpha_generator.py
    python3 residual_alpha_generator.py --region HKG --universe TOP500 --delay 1
"""

import argparse
import sys

from ace_lib import get_datafields, start_session
from alpha_choices import pick_region_universe_delay
from alpha_pipeline import alphas_dir_for, build_settings, write_alphas_jsonl

BACKFILL_DAYS = 63
WINSORIZE_STD = 4.0
SIZE_MEAN_DAYS = 21
REGRESSION_DAYS = 252
GROUP = "sector"

FUNDAMENTAL_KEYWORDS = [
    "asset", "equity", "income", "earning", "profit", "revenue",
    "sales", "cash_flow", "fcf", "roe", "roa", "roic",
    "margin", "yield", "debt", "book", "ebit", "ebitda",
    "dividend", "capex", "inventory", "liabilit",
]

ANALYST_KEYWORDS = [
    "analyst", "rating", "target", "estimate", "eps_est",
    "consensus", "recommendation", "upgrad", "downgrad",
]

SENTIMENT_KEYWORDS = [
    "sentiment", "news", "buzz", "social", "twitter", "reddit",
]

SEARCH_TERMS = {
    "fundamental": ["fundamental", "income", "asset", "equity", "earnings",
                    "profit", "revenue", "cash_flow", "margin", "book_value"],
    "analyst":     ["analyst", "rating", "estimate", "target", "consensus"],
    "sentiment":   ["sentiment", "news", "social", "buzz"],
}

PRICE_SIGNALS = [
    "returns",
    "ts_returns(close, 5)",
    "ts_returns(close, 20)",
    "ts_returns(close, 60)",
    "close / ts_mean(close, 20)",
    "close / ts_mean(close, 60)",
    "close / ts_mean(close, 200)",
    "volume / adv20",
    "ts_std_dev(returns, 20)",
    "ts_std_dev(returns, 60)",
]


def classify_field(row) -> str | None:
    field_id = str(row.get("id", row.get("name", ""))).lower()
    desc = str(row.get("description", "")).lower()
    category = str(row.get("category", "")).lower()
    dataset = str(row.get("dataset_id", row.get("datasetId", ""))).lower()
    text = f"{field_id} {desc}"

    if field_id.startswith("fnd"):
        return "fundamental"
    if field_id.startswith("anl"):
        return "analyst"
    if field_id.startswith(("nws", "scl")):
        return "sentiment"

    if "fundamental" in category or "fundamental" in dataset:
        return "fundamental"
    if "analyst" in category or "analyst" in dataset:
        return "analyst"
    if any(k in category or k in dataset for k in ("sentiment", "news", "social")):
        return "sentiment"

    if any(kw in text for kw in ANALYST_KEYWORDS):
        return "analyst"
    if any(kw in text for kw in SENTIMENT_KEYWORDS):
        return "sentiment"
    if any(kw in text for kw in FUNDAMENTAL_KEYWORDS):
        return "fundamental"

    return None


def discover_fields(s, region: str, universe: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {"fundamental": [], "analyst": [], "sentiment": []}
    seen: set[str] = set()

    for category, terms in SEARCH_TERMS.items():
        for term in terms:
            print(f"  Searching '{term}'...")
            df = get_datafields(s, region=region, universe=universe, search=term)
            if df.empty:
                continue
            for _, row in df.iterrows():
                field_id = str(row.get("id", row.get("name", "")))
                if not field_id or field_id in seen:
                    continue
                classified = classify_field(row)
                if classified and classified in fields:
                    fields[classified].append(field_id)
                    seen.add(field_id)

    for cat, lst in fields.items():
        print(f"  {cat:12s} ({len(lst):>4}): {lst[:5]}{'...' if len(lst) > 5 else ''}")
    return fields


def build_expression(data_expr: str) -> str:
    return (
        f"data = winsorize(ts_backfill({data_expr}, {BACKFILL_DAYS}), std={WINSORIZE_STD});\n"
        f"data_gpm = group_mean(data, log(ts_mean(cap, {SIZE_MEAN_DAYS})), {GROUP});\n"
        f"ts_regression(data, data_gpm, {REGRESSION_DAYS}, rettype=0)"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate residual alpha combinations")
    parser.add_argument("--region", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--delay", type=int, default=None)
    args = parser.parse_args()

    s = start_session()

    region, universe, delay = pick_region_universe_delay(
        s,
        default_region=args.region or "HKG",
        default_universe=args.universe or "TOP500",
        default_delay=args.delay if args.delay is not None else 1,
    )

    settings = build_settings(region=region, universe=universe, delay=delay)
    output_dir = alphas_dir_for("residual", settings)

    print("Discovering datafields from WQ Brain...")
    fields = discover_fields(s, region, universe)
    fields["price"] = list(PRICE_SIGNALS)

    total = sum(len(v) for v in fields.values())
    if total == 0:
        print("ERROR: No usable data signals found.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Residual Alpha Generator (sector/size-neutralized)")
    print(f"{'='*60}")
    print(f"  Region / Universe : {region} / {universe}  delay={delay}")
    for cat, lst in fields.items():
        print(f"  {cat:12s} signals : {len(lst):>4}")
    print(f"  Total alphas      : {total:,}")
    print(f"  Template          : winsorize → group_mean → ts_regression")
    print(f"  Output dir        : {output_dir}/")
    print(f"{'='*60}\n")

    print("Generating alpha expressions...")
    all_exprs: list[str] = []
    for category, data_exprs in fields.items():
        if not data_exprs:
            print(f"  {category:12s}: SKIPPED (no signals)")
            continue
        exprs = [build_expression(d) for d in data_exprs]
        all_exprs.extend(exprs)
        print(f"  {category:12s}: {len(exprs):>6,} alphas")

    output_path = f"{output_dir}/residual.jsonl"
    grand_total = write_alphas_jsonl(iter(all_exprs), output_path)

    print(f"\nDone! {grand_total:,} alphas → {output_path}")
    if all_exprs:
        print(f"\nExample expression:\n{all_exprs[0]}")


if __name__ == "__main__":
    main()
