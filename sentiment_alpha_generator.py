"""
Program 1: Sentiment Alpha Generator
=====================================
Fetches sentiment datafields from the API, classifies them as
positive/negative, then generates all alpha expression combinations
using curated operators and saves them for simulation.

Usage:
    python3 sentiment_alpha_generator.py

Output:
    generated_alphas.jsonl  — one alpha dict per line (compact)
"""

import itertools
import json
import os
import sys

import pandas as pd

from ace_lib import (
    generate_alpha,
    get_datafields,
    get_operators,
    start_session,
)

# ── curated operators (only ones that make sense for sentiment template) ──────
#
# Backfill ops: smooth/fill sparse sentiment data before ranking
#   ts_backfill  — fill NaN gaps in sentiment data (essential)
#   ts_mean      — simple rolling average, basic smoother
#   ts_decay_linear — weighted avg favoring recent days, great for sentiment
#   ts_sum       — accumulate sentiment signal over window
#   ts_av_diff   — deviation from mean (x - ts_mean(x,d)), captures surprise
#
BACKFILL_OPS = [
    "ts_backfill",
    "ts_mean",
    "ts_decay_linear",
    "ts_sum",
    "ts_av_diff",
]

# Compare ops: compute difference between positive vs negative sentiment
#   subtract — direct difference (pos - neg), preserves magnitude
#   divide   — ratio (pos / neg), captures relative strength
#
COMPARE_OPS = [
    "subtract",
    "divide",
]

# Time series ops: final transformation on the sentiment difference signal
#   ts_delta        — momentum/change in sentiment diff
#   ts_zscore       — how unusual is current diff vs recent history
#   ts_rank         — where does current diff rank in recent window
#   ts_decay_linear — smoothed trend of sentiment diff
#   ts_mean         — average sentiment diff over window
#   ts_scale        — normalize to 0-1 range
#   ts_ir           — information ratio (mean/stddev), signal-to-noise
#
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

OUTPUT_DIR = "generated_alphas"

# ── sentiment classification keywords ────────────────────────────────────────

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
    """Classify a datafield as 'positive', 'negative', or None (ambiguous)."""
    text = (field_id + " " + str(field_desc)).lower()

    pos_score = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg_score = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)

    if pos_score > neg_score:
        return "positive"
    if neg_score > pos_score:
        return "negative"
    return None


def get_sentiment_fields(s) -> tuple[list[str], list[str]]:
    """Fetch sentiment datafields and split into positive / negative lists."""
    print("Fetching sentiment datafields...")
    df = get_datafields(s, search="sentiment")

    if df.empty:
        print("WARNING: No sentiment datafields found")
        return [], []

    positive, negative = [], []

    for _, row in df.iterrows():
        field_id = row.get("id", row.get("name", ""))
        desc = row.get("description", "")
        classification = classify_sentiment(str(field_id), str(desc))

        if classification == "positive":
            positive.append(str(field_id))
        elif classification == "negative":
            negative.append(str(field_id))
        else:
            # ambiguous fields go into both lists so no data is lost
            positive.append(str(field_id))
            negative.append(str(field_id))

    print(f"  Positive sentiment fields ({len(positive)}): {positive[:5]}{'...' if len(positive) > 5 else ''}")
    print(f"  Negative sentiment fields ({len(negative)}): {negative[:5]}{'...' if len(negative) > 5 else ''}")
    return positive, negative


def get_filtered_operators(s) -> tuple[list[str], list[str], list[str]]:
    """Validate curated operators exist on the platform."""
    print("Validating operators against platform...")
    df = get_operators(s)
    all_ops = set(df["name"].unique()) if "name" in df.columns else set(df.iloc[:, 0].unique())

    backfill = [op for op in BACKFILL_OPS if op in all_ops]
    compare = [op for op in COMPARE_OPS if op in all_ops]
    ts = [op for op in TIME_SERIES_OPS if op in all_ops]

    missing_bf = set(BACKFILL_OPS) - set(backfill)
    missing_cmp = set(COMPARE_OPS) - set(compare)
    missing_ts = set(TIME_SERIES_OPS) - set(ts)

    if missing_bf:
        print(f"  WARNING: backfill ops not on platform: {missing_bf}")
    if missing_cmp:
        print(f"  WARNING: compare ops not on platform: {missing_cmp}")
    if missing_ts:
        print(f"  WARNING: ts ops not on platform: {missing_ts}")

    print(f"  Backfill ops  ({len(backfill)}): {backfill}")
    print(f"  Compare ops   ({len(compare)}): {compare}")
    print(f"  TS ops        ({len(ts)}): {ts}")
    return backfill, compare, ts


def build_expression(
    pos_field: str,
    neg_field: str,
    backfill_op: str,
    compare_op: str,
    ts_op: str,
    days_rank: int,
    days_final: int,
) -> str:
    """Build an alpha expression from the template."""
    lines = [
        f"positive_sentiment = rank({backfill_op}({pos_field}, {days_rank}));",
        f"negative_sentiment = rank({backfill_op}({neg_field}, {days_rank}));",
        f"sentiment_difference = {compare_op}(positive_sentiment, negative_sentiment);",
        f"{ts_op}(sentiment_difference, {days_final})",
    ]
    return "\n".join(lines)


def main():
    s = start_session()

    # 1. Get sentiment fields
    positive_fields, negative_fields = get_sentiment_fields(s)
    if not positive_fields or not negative_fields:
        print("ERROR: Could not find both positive and negative sentiment fields. Exiting.")
        sys.exit(1)

    # 2. Validate operators
    backfill_ops, compare_ops, ts_ops = get_filtered_operators(s)
    if not backfill_ops or not compare_ops or not ts_ops:
        print("ERROR: Missing operator categories. Exiting.")
        sys.exit(1)

    # 3. Count combinations
    total = (
        len(positive_fields)
        * len(negative_fields)
        * len(backfill_ops)
        * len(compare_ops)
        * len(ts_ops)
        * len(DAYS)  # days_rank
        * len(DAYS)  # days_final
    )

    # Number of files = len(DAYS) * len(compare_ops)
    num_files = len(DAYS) * len(compare_ops)
    alphas_per_file = total // num_files if num_files > 0 else total

    print(f"\n{'='*60}")
    print(f"  Sentiment Alpha Generator")
    print(f"{'='*60}")
    print(f"  Positive fields : {len(positive_fields)}")
    print(f"  Negative fields : {len(negative_fields)}")
    print(f"  Backfill ops    : {len(backfill_ops)}")
    print(f"  Compare ops     : {len(compare_ops)}")
    print(f"  TS ops          : {len(ts_ops)}")
    print(f"  Days variants   : {DAYS}")
    print(f"  Total alphas    : {total:,}")
    print(f"  Output files    : {num_files} files in {OUTPUT_DIR}/")
    print(f"  ~Alphas/file    : {alphas_per_file:,}")
    print(f"{'='*60}\n")

    # 4. Generate and write split by (days_rank, compare_op)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating alpha expressions...")
    grand_total = 0
    example_expr = None

    for d_rank in DAYS:
        for cmp_op in compare_ops:
            filename = f"d{d_rank:03d}_{cmp_op}.jsonl"
            filepath = os.path.join(OUTPUT_DIR, filename)
            file_count = 0

            with open(filepath, "w") as f:
                for pos, neg, bf_op, ts_op, d_final in itertools.product(
                    positive_fields,
                    negative_fields,
                    backfill_ops,
                    ts_ops,
                    DAYS,
                ):
                    expr = build_expression(pos, neg, bf_op, cmp_op, ts_op, d_rank, d_final)
                    alpha = generate_alpha(regular=expr)
                    f.write(json.dumps(alpha) + "\n")
                    file_count += 1

                    if example_expr is None:
                        example_expr = expr

            grand_total += file_count
            print(f"  {filename:30s} -> {file_count:>8,} alphas")

    print(f"\nDone! {grand_total:,} alphas across {num_files} files in {OUTPUT_DIR}/")

    # 5. Show example
    if example_expr:
        print(f"\nExample expression (first alpha):\n")
        print(example_expr)

    print(f"\nNext: python3 sentiment_alpha_simulator.py --input {OUTPUT_DIR}/d005_subtract.jsonl")


if __name__ == "__main__":
    main()
