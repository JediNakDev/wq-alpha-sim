"""
Fundamental Ratio Alpha Generator
===================================
Hypothesis: A company's performance ratio (profitability / size) trending up
implies improving capital efficiency, which should lead stock price higher.

Template:
    profit_rank = group_rank(<profit_field>, <group>);
    size_rank = group_rank(<size_field>, <group>);
    performance_ratio = profit_rank / (size_rank + 1);
    <ts_op>(performance_ratio, <days>)

Usage:
    python3 fundamental_alpha_generator.py
    python3 fundamental_alpha_generator.py --region HKG --universe TOP500 --delay 1
"""

import argparse
import itertools
import sys

from ace_lib import get_datafields, get_operators, start_session
from alpha_choices import pick_region_universe_delay
from alpha_pipeline import alphas_dir_for, build_settings, write_alphas_jsonl

# ── Operator sets ─────────────────────────────────────────────

TIME_SERIES_OPS = [
    "ts_delta",
    "ts_zscore",
    "ts_rank",
    "ts_decay_linear",
    "ts_mean",
    "ts_scale",
    "ts_ir",
]

GROUPS = ["industry", "sector", "subindustry"]
DAYS = [5, 20, 60, 250]

# ── Field classification keywords ────────────────────────────

PROFIT_KEYWORDS = [
    "income", "earning", "profit", "ebit", "ebitda", "net_income",
    "operating_income", "gross_profit", "pretax_income", "revenue",
    "sales", "return_on", "roe", "roa", "roic", "roi", "eps",
    "cash_flow", "free_cash_flow", "fcf", "margin", "yield",
    "dividend", "operating_cash", "net_profit",
]

SIZE_KEYWORDS = [
    "market_cap", "total_asset", "total_equity", "book_value",
    "enterprise_value", "ev", "shares_outstanding", "share_outstanding",
    "total_debt", "total_liabilit", "capitalization", "assets",
    "equity", "book", "mcap", "mktcap", "market_value",
    "total_revenue",
]


def classify_field(field_id: str, field_desc: str) -> str | None:
    text = (field_id + " " + str(field_desc)).lower()
    profit_score = sum(1 for kw in PROFIT_KEYWORDS if kw in text)
    size_score = sum(1 for kw in SIZE_KEYWORDS if kw in text)
    if profit_score > 0 and profit_score > size_score:
        return "profit"
    if size_score > 0 and size_score > profit_score:
        return "size"
    return None


def get_fundamental_fields(s, region: str, universe: str) -> tuple[list[str], list[str]]:
    profit_fields, size_fields = [], []
    for term in ["fundamental", "financial", "income", "asset", "equity",
                 "earnings", "profit", "revenue", "market_cap", "book_value"]:
        print(f"  Searching: '{term}'...")
        df = get_datafields(s, region=region, universe=universe, search=term)
        if df.empty:
            continue
        for _, row in df.iterrows():
            field_id = str(row.get("id", row.get("name", "")))
            desc = str(row.get("description", ""))
            cls = classify_field(field_id, desc)
            if cls == "profit" and field_id not in profit_fields:
                profit_fields.append(field_id)
            elif cls == "size" and field_id not in size_fields:
                size_fields.append(field_id)

    print(f"  Profit fields ({len(profit_fields)}): {profit_fields[:5]}{'...' if len(profit_fields) > 5 else ''}")
    print(f"  Size fields   ({len(size_fields)}): {size_fields[:5]}{'...' if len(size_fields) > 5 else ''}")
    return profit_fields, size_fields


def get_filtered_operators(s) -> list[str]:
    print("Validating operators...")
    df = get_operators(s)
    all_ops = set(df["name"].unique()) if "name" in df.columns else set(df.iloc[:, 0].unique())
    ts = [op for op in TIME_SERIES_OPS if op in all_ops]
    missing = set(TIME_SERIES_OPS) - set(ts)
    if missing:
        print(f"  WARNING: TS ops not on platform: {missing}")
    print(f"  TS ops ({len(ts)}): {ts}")
    return ts


def build_expression(profit_field: str, size_field: str, group: str, ts_op: str, days: int) -> str:
    return "\n".join([
        f"profit_rank = group_rank({profit_field}, {group});",
        f"size_rank = group_rank({size_field}, {group});",
        f"performance_ratio = profit_rank / (size_rank + 1);",
        f"{ts_op}(performance_ratio, {days})",
    ])


def main():
    parser = argparse.ArgumentParser(description="Generate fundamental ratio alpha combinations")
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
    output_dir = alphas_dir_for("fundamental", settings)

    print("Fetching fundamental datafields...")
    profit_fields, size_fields = get_fundamental_fields(s, region, universe)
    if not profit_fields or not size_fields:
        print("ERROR: Could not find both profit and size fields.")
        sys.exit(1)

    ts_ops = get_filtered_operators(s)
    if not ts_ops:
        print("ERROR: No valid time-series operators found.")
        sys.exit(1)

    total = len(profit_fields) * len(size_fields) * len(GROUPS) * len(ts_ops) * len(DAYS)
    num_files = len(DAYS) * len(ts_ops)

    print(f"\n{'='*60}")
    print(f"  Fundamental Ratio Alpha Generator")
    print(f"{'='*60}")
    print(f"  Region / Universe : {region} / {universe}  delay={delay}")
    print(f"  Profit fields     : {len(profit_fields)}")
    print(f"  Size fields       : {len(size_fields)}")
    print(f"  Groups            : {GROUPS}")
    print(f"  TS ops            : {len(ts_ops)}")
    print(f"  Days              : {DAYS}")
    print(f"  Total alphas      : {total:,}")
    print(f"  Output files      : {num_files} files in {output_dir}/")
    print(f"{'='*60}\n")

    print("Generating alpha expressions...")
    grand_total = 0
    example_expr = None

    for days in DAYS:
        for ts_op in ts_ops:
            output_path = f"{output_dir}/d{days:03d}_{ts_op}.jsonl"

            def _expressions(days=days, ts_op=ts_op):
                for profit, size, group in itertools.product(profit_fields, size_fields, GROUPS):
                    yield build_expression(profit, size, group, ts_op, days)

            exprs = list(_expressions())
            count = write_alphas_jsonl(iter(exprs), output_path)
            grand_total += count
            print(f"  {output_path:60s} -> {count:>8,} alphas")

            if example_expr is None and exprs:
                example_expr = exprs[0]

    print(f"\nDone! {grand_total:,} alphas in {num_files} files → {output_dir}/")
    if example_expr:
        print(f"\nExample expression:\n{example_expr}")
    print(f"\nNext: python3 fundamental_alpha_simulator.py --input {output_dir}/d005_ts_delta.jsonl")


if __name__ == "__main__":
    main()
