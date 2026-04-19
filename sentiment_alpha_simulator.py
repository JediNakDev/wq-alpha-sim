"""
Sentiment Alpha Simulator
=========================
Region and universe are inferred from the input file path.

Usage:
    python3 sentiment_alpha_simulator.py --input alphas/sentiment_usa_top3000/d005_subtract.jsonl
    python3 sentiment_alpha_simulator.py --input alphas/sentiment_usa_top3000/d005_subtract.jsonl --fresh
"""

import argparse

from alpha_pipeline import infer_settings_from_path, results_dir_for, simulate_alphas

DESCRIPTION = (
    "Idea: This alpha seeks to capture the directional sentiment momentum of a "
    "security by comparing how its positive and negative sentiment signals evolve "
    "over time. The core thesis is that stocks where positive sentiment is "
    "consistently outpacing negative sentiment on a relative basis are likely to "
    "continue attracting investor interest and generate positive returns. By "
    "trading on the spread between ranked sentiment measures, the alpha attempts "
    "to isolate securities where the sentiment backdrop is improving in a "
    "sustained, cross-sectionally significant way. The final time-series trade "
    "operator introduces a mean-reversion or momentum overlay on top of that "
    "spread, depending on the operator chosen, to time entries and exits more "
    "precisely.\n"
    "Rationale for data used: The alpha uses two sentiment signals — positive "
    "sentiment and negative sentiment — which typically derive from natural "
    "language processing of news articles, analyst reports, social media, or "
    "earnings call transcripts. Positive sentiment captures the frequency or "
    "intensity of optimistic language surrounding a stock, while negative "
    "sentiment does the same for pessimistic language. Using both independently "
    "rather than a single net sentiment figure allows the alpha to treat the two "
    "dimensions asymmetrically, which is important because markets are known to "
    "react differently to good versus bad news. The days parameter controls the "
    "lookback window for the time-series operator applied to each sentiment "
    "stream, allowing the alpha to smooth out noise and focus on sustained "
    "sentiment trends rather than transient spikes.\n"
    "Rationale for operators used: The ts operator applied to each raw sentiment "
    "stream first aggregates or smooths the signal over a rolling window of days, "
    "reducing noise and anchoring the signal to a persistent trend rather than a "
    "single observation. The rank operator is then applied cross-sectionally to "
    "both the smoothed positive and negative sentiment values, normalizing each "
    "across the universe to remove level differences between stocks and ensure "
    "comparability regardless of absolute sentiment magnitude. The compare op "
    "computes the difference or ratio between the two ranked signals, producing a "
    "single spread that reflects whether positive sentiment dominates negative "
    "sentiment on a relative, universe-adjusted basis. Finally, ts trade operator "
    "applies a time-series transformation over days to this spread, which "
    "depending on the specific operator could implement momentum, mean-reversion, "
    "or decay weighting to generate the final tradeable signal."
)


def main():
    parser = argparse.ArgumentParser(description="Simulate sentiment alphas")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--retry-wait", type=int, default=60)
    args = parser.parse_args()

    settings = infer_settings_from_path(
        args.input, delay=args.delay, neutralization="SLOW_AND_FAST", truncation=0.08
    )
    results_dir = results_dir_for("sentiment", settings)

    simulate_alphas(
        input_file=args.input,
        settings=settings,
        description=DESCRIPTION,
        category="SENTIMENT",
        results_dir=results_dir,
        threads=args.threads,
        batch_size=args.batch_size,
        retry_wait=args.retry_wait,
        fresh=args.fresh,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
