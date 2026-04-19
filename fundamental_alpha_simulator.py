"""
Fundamental Ratio Alpha Simulator
==================================
Region and universe are inferred from the input file path.

Usage:
    python3 fundamental_alpha_simulator.py --input alphas/fundamental_hkg_top500/d005_ts_delta.jsonl
    python3 fundamental_alpha_simulator.py --input alphas/fundamental_hkg_top500/d005_ts_delta.jsonl --fresh
"""

import argparse

from alpha_pipeline import infer_settings_from_path, results_dir_for, simulate_alphas

DESCRIPTION = (
    "Idea: This alpha captures the trend in a company's performance ratio — "
    "profitability relative to size — ranked within its peer group. The "
    "hypothesis is that if a company's fundamentals-based performance ratio "
    "is increasing over time, its stock price is likely to follow. By using "
    "group_rank on both the profit metric and the size metric within the same "
    "industry/sector group, the alpha normalizes for cross-sectional "
    "differences and isolates companies that are improving their profitability "
    "relative to their size faster than their peers. The time-series operator "
    "then captures the momentum or trend of this ratio, turning the signal "
    "into a tradeable alpha.\n"
    "Rationale for data used: The alpha uses two categories of fundamental "
    "data — a profit field and a size field. The profit field (such as net "
    "income, operating income, EBIT, EBITDA, earnings per share, return on "
    "equity, or free cash flow) measures how effectively a company converts "
    "its resources into earnings. These metrics are chosen because sustained "
    "profitability improvement is one of the strongest fundamental drivers of "
    "long-term equity returns. The size field (such as market capitalization, "
    "total assets, book value, enterprise value, or shareholders equity) "
    "captures the scale of the business and serves as the denominator in the "
    "performance ratio. Normalizing profit by size is essential because a "
    "$1B company earning $100M is fundamentally different from a $100B "
    "company earning the same amount — the smaller firm is deploying capital "
    "far more efficiently.\n"
    "Rationale for operators used: group_rank is applied independently to "
    "both the profit field and the size field within the same peer group "
    "(industry, sector, or subindustry). This cross-sectional ranking "
    "normalizes each metric to a 0-1 percentile score within comparable "
    "firms, removing structural differences between industries. The ratio "
    "profit_rank / (size_rank + 1) produces a score where companies with "
    "high profitability rank relative to their size rank score highest; "
    "the +1 prevents division by zero. The final time-series operator "
    "captures the trend or momentum of this performance ratio over the "
    "specified window."
)


def main():
    parser = argparse.ArgumentParser(description="Simulate fundamental ratio alphas")
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
    results_dir = results_dir_for("fundamental", settings)

    simulate_alphas(
        input_file=args.input,
        settings=settings,
        description=DESCRIPTION,
        category="FUNDAMENTAL",
        results_dir=results_dir,
        threads=args.threads,
        batch_size=args.batch_size,
        retry_wait=args.retry_wait,
        fresh=args.fresh,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
