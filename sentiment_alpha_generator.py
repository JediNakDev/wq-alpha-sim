"""
Sentiment Alpha Generator
=========================
Fetches sentiment datafields, classifies them as positive/negative,
and generates all alpha expression combinations.

Usage:
    python3 sentiment_alpha_generator.py
    python3 sentiment_alpha_generator.py --region USA --universe TOP3000 --delay 1
"""

import argparse
import itertools
import sys

from ace_lib import get_datafields, get_operators, start_session
from alpha_choices import pick_region_universe_delay
from alpha_pipeline import alphas_dir_for, build_settings, write_alphas_jsonl

# ── Curated operators ─────────────────────────────────────────

BACKFILL_OPS = [
    "ts_backfill",
    "ts_mean",
    "ts_decay_linear",
    "ts_sum",
    "ts_av_diff",
]

COMPARE_OPS = [
    "subtract",
    "divide",
]

TIME_SERIES_OPS = [
    "ts_delta",
    "ts_zscore",
    "ts_rank",
    "ts_decay_linear",
    "ts_mean",
    "ts_scale",
    "ts_ir",
]

DAYS = [5, 20, 60, 250]

# ── Sentiment classification ──────────────────────────────────

POSITIVE_KEYWORDS = [
    "bull", "buy", "upgrad", "positive", "optimis", "long", "strong",
    "outperform", "overweight", "accumulat", "recommend_buy",
    "favorable", "raise", "beat", "above", "up", "growth",
    "improve", "increas", "higher", "gain", "recover", "advanc",
    "boost", "surpass", "exceed", "profit", "revenue_up",
    "consensus_above", "estimate_above", "surprise_positive",
]

NEGATIVE_KEYWORDS = [
    "bear", "sell", "downgrad", "negative", "pessimis", "short", "weak",
    "underperform", "underweight", "distribut", "recommend_sell",
    "unfavorable", "lower", "miss", "below", "down", "decline",
    "worsen", "decreas", "loss", "drop", "fall", "cut",
    "disappoint", "deficit", "risk", "warn",
    "consensus_below", "estimate_below", "surprise_negative",
]


def classify_sentiment(field_id: str, field_desc: str) -> str | None:
    text = (field_id + " " + str(field_desc)).lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return None


def get_sentiment_fields(s, region: str, universe: str) -> tuple[list[str], list[str]]:
    print("Fetching sentiment datafields...")
    df = get_datafields(s, region=region, universe=universe, search="sentiment")
    if df.empty:
        print("WARNING: No sentiment datafields found")
        return [], []

    positive, negative = [], []
    for _, row in df.iterrows():
        field_id = row.get("id", row.get("name", ""))
        desc = row.get("description", "")
        cls = classify_sentiment(str(field_id), str(desc))
        if cls == "positive":
            positive.append(str(field_id))
        elif cls == "negative":
            negative.append(str(field_id))
        else:
            positive.append(str(field_id))
            negative.append(str(field_id))

    print(f"  Positive ({len(positive)}): {positive[:5]}{'...' if len(positive) > 5 else ''}")
    print(f"  Negative ({len(negative)}): {negative[:5]}{'...' if len(negative) > 5 else ''}")
    return positive, negative


def get_filtered_operators(s) -> tuple[list[str], list[str], list[str]]:
    print("Validating operators against platform...")
    df = get_operators(s)
    all_ops = set(df["name"].unique()) if "name" in df.columns else set(df.iloc[:, 0].unique())

    backfill = [op for op in BACKFILL_OPS if op in all_ops]
    compare = [op for op in COMPARE_OPS if op in all_ops]
    ts = [op for op in TIME_SERIES_OPS if op in all_ops]

    missing = (set(BACKFILL_OPS) - set(backfill)) | (set(COMPARE_OPS) - set(compare)) | (set(TIME_SERIES_OPS) - set(ts))
    if missing:
        print(f"  WARNING: ops not on platform: {missing}")

    print(f"  Backfill ({len(backfill)}): {backfill}")
    print(f"  Compare  ({len(compare)}): {compare}")
    print(f"  TS       ({len(ts)}): {ts}")
    return backfill, compare, ts


def build_expression(
    pos_field: str, neg_field: str,
    backfill_op: str, compare_op: str, ts_op: str,
    days_rank: int, days_final: int,
) -> str:
    return "\n".join([
        f"positive_sentiment = rank({backfill_op}({pos_field}, {days_rank}));",
        f"negative_sentiment = rank({backfill_op}({neg_field}, {days_rank}));",
        f"sentiment_difference = {compare_op}(positive_sentiment, negative_sentiment);",
        f"group_neutralize({ts_op}(sentiment_difference, {days_final}), industry)",
    ])


def main():
    parser = argparse.ArgumentParser(description="Generate sentiment alpha combinations")
    parser.add_argument("--region", default=None, help="Region (e.g. USA, HKG)")
    parser.add_argument("--universe", default=None, help="Universe (e.g. TOP3000, TOP500)")
    parser.add_argument("--delay", type=int, default=None, help="Delay (0 or 1)")
    args = parser.parse_args()

    s = start_session()

    region, universe, delay = pick_region_universe_delay(
        s,
        default_region=args.region or "USA",
        default_universe=args.universe or "TOP3000",
        default_delay=args.delay if args.delay is not None else 1,
    )

    settings = build_settings(region=region, universe=universe, delay=delay)
    output_dir = alphas_dir_for("sentiment", settings)

    positive_fields, negative_fields = get_sentiment_fields(s, region, universe)
    if not positive_fields or not negative_fields:
        print("ERROR: Could not find both positive and negative sentiment fields.")
        sys.exit(1)

    backfill_ops, compare_ops, ts_ops = get_filtered_operators(s)
    if not backfill_ops or not compare_ops or not ts_ops:
        print("ERROR: Missing operator categories.")
        sys.exit(1)

    total = (
        len(positive_fields) * len(negative_fields)
        * len(backfill_ops) * len(compare_ops) * len(ts_ops)
        * len(DAYS) * len(DAYS)
    )
    num_files = len(DAYS) * len(compare_ops)

    print(f"\n{'='*60}")
    print(f"  Sentiment Alpha Generator")
    print(f"{'='*60}")
    print(f"  Region / Universe : {region} / {universe}  delay={delay}")
    print(f"  Positive fields   : {len(positive_fields)}")
    print(f"  Negative fields   : {len(negative_fields)}")
    print(f"  Backfill ops      : {len(backfill_ops)}")
    print(f"  Compare ops       : {len(compare_ops)}")
    print(f"  TS ops            : {len(ts_ops)}")
    print(f"  Days              : {DAYS}")
    print(f"  Total alphas      : {total:,}")
    print(f"  Output files      : {num_files} files in {output_dir}/")
    print(f"{'='*60}\n")

    print("Generating alpha expressions...")
    grand_total = 0
    example_expr = None

    for d_rank in DAYS:
        for cmp_op in compare_ops:
            output_path = f"{output_dir}/d{d_rank:03d}_{cmp_op}.jsonl"

            def _expressions(d_rank=d_rank, cmp_op=cmp_op):
                for pos, neg, bf_op, ts_op, d_final in itertools.product(
                    positive_fields, negative_fields, backfill_ops, ts_ops, DAYS
                ):
                    yield build_expression(pos, neg, bf_op, cmp_op, ts_op, d_rank, d_final)

            exprs = list(_expressions())
            count = write_alphas_jsonl(iter(exprs), output_path)
            grand_total += count
            print(f"  {output_path:55s} -> {count:>8,} alphas")

            if example_expr is None and exprs:
                example_expr = exprs[0]

    print(f"\nDone! {grand_total:,} alphas in {num_files} files → {output_dir}/")
    if example_expr:
        print(f"\nExample expression:\n{example_expr}")
    print(f"\nNext: python3 sentiment_alpha_simulator.py --input {output_dir}/d005_subtract.jsonl")


if __name__ == "__main__":
    main()
