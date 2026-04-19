"""
Residual Alpha Simulator
=========================
Region and universe are inferred from the input file path.

Usage:
    python3 residual_alpha_simulator.py --input alphas/residual_hkg_top500/residual.jsonl
    python3 residual_alpha_simulator.py --input alphas/residual_hkg_top500/residual.jsonl --fresh
"""

import argparse

from alpha_pipeline import infer_settings_from_path, results_dir_for, simulate_alphas

DESCRIPTION = (
    "Idea: This alpha isolates the idiosyncratic component of a raw slow-moving "
    "signal by regressing it against its sector/size peer-group mean. The core "
    "thesis is that the residual after removing group-level co-movement represents "
    "genuine company-specific information — alpha that cannot be explained by "
    "industry trends or market-cap effects. Signals with stronger idiosyncratic "
    "content relative to their peer group should predict stock-specific returns.\n"
    "Rationale for data used: Four categories of slow-moving signals are used — "
    "fundamental (quarterly/annual accounting data), analyst (ratings, estimates, "
    "target prices), sentiment (news / social buzz), and price-derived signals "
    "(returns, volume ratios). All are winsorized to remove outliers and "
    "backfilled over 63 trading days to handle sparse reporting schedules. "
    "The group peer mean uses log(market cap) as the size covariate within "
    "the sector grouping to ensure the benchmark captures both industry and "
    "size effects simultaneously.\n"
    "Rationale for operators used: ts_backfill fills gaps caused by infrequent "
    "reporting. winsorize(std=4.0) clips extreme values that would distort the "
    "regression. group_mean with log(cap) and sector partitions the universe so "
    "the peer benchmark is computed within economically comparable firms. "
    "ts_regression over 252 days projects the signal onto its peer mean and "
    "returns the residual (rettype=0), which is the idiosyncratic component "
    "stripped of systematic group variation."
)


def main():
    parser = argparse.ArgumentParser(description="Simulate residual alphas")
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
    results_dir = results_dir_for("residual", settings)

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
