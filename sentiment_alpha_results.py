"""
Program 3: Sentiment Alpha Results Extractor
==============================================
Reads sentiment_results/submissions.jsonl, checks which simulations
have completed, fetches their results (stats, checks), and outputs
a consolidated CSV.

Usage:
    python3 sentiment_alpha_results.py                      # fetch all completed results
    python3 sentiment_alpha_results.py --poll-pending        # also resolve still-running sims
    python3 sentiment_alpha_results.py --check-submission    # run submission eligibility checks
    python3 sentiment_alpha_results.py --summary            # just print stats, no fetching

Output:
    sentiment_results/results.csv  — main results table
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing.pool import ThreadPool

import pandas as pd

from ace_lib import (
    brain_api_url,
    check_session_and_relogin,
    get_simulation_result_json,
    get_check_submission,
    check_self_corr_test,
    check_prod_corr_test,
    start_session,
    SingleSession,
)

RESULTS_DIR = "sentiment_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")


def find_submission_files() -> list[str]:
    """Find all sub_*.jsonl files in the results directory."""
    if not os.path.exists(RESULTS_DIR):
        return []
    files = sorted(
        os.path.join(RESULTS_DIR, f)
        for f in os.listdir(RESULTS_DIR)
        if f.startswith("sub_") and f.endswith(".jsonl")
    )
    return files


def load_submissions() -> tuple[list[dict], list[str]]:
    """Load all submission records from all sub_*.jsonl files."""
    files = find_submission_files()
    if not files:
        print(f"ERROR: No submission files (sub_*.jsonl) found in {RESULTS_DIR}/")
        print("Run sentiment_alpha_simulator.py first.")
        sys.exit(1)

    print(f"Found {len(files)} submission files:")
    for f in files:
        print(f"  {f}")

    entries = []
    for filepath in files:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entry["_source_file"] = filepath
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
    return entries, files


def save_submissions(submissions: list[dict]):
    """Rewrite submissions back to their source files."""
    # Group by source file
    by_file: dict[str, list[dict]] = {}
    for entry in submissions:
        source = entry.pop("_source_file", None)
        if source:
            by_file.setdefault(source, []).append(entry)

    for filepath, entries in by_file.items():
        with open(filepath, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")


def poll_pending(s: SingleSession, submission: dict) -> dict:
    """Poll a 'submitted' simulation to see if it has completed."""
    progress_url = submission.get("progress_url")
    if not progress_url:
        return submission

    try:
        resp = s.get(progress_url)
        if resp.status_code // 100 != 2:
            return submission

        retry_after = resp.headers.get("Retry-After", 0)
        if retry_after != 0 and str(retry_after) != "0":
            return submission  # still running

        data = resp.json()
        alpha_id = data.get("alpha", None)

        if data.get("status") == "ERROR" or alpha_id is None:
            submission["status"] = "sim_failed"
            submission["error"] = json.dumps(data)[:500]
        else:
            submission["status"] = "completed"
            submission["alpha_id"] = alpha_id
            submission["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        pass  # leave as-is, will retry next run

    return submission


def fetch_alpha_result(s: SingleSession, alpha_id: str, expression: str, index: int,
                       check_submission: bool = False,
                       check_self_corr: bool = False,
                       check_prod_corr: bool = False) -> dict:
    """Fetch full result for a single completed alpha."""
    try:
        s = check_session_and_relogin(s)
        result = get_simulation_result_json(s, alpha_id)

        if not result:
            return {"alpha_id": alpha_id, "index": index, "status": "fetch_failed"}

        is_data = result.get("is", {})
        checks = is_data.pop("checks", [])

        entry = {
            "alpha_id": alpha_id,
            "index": index,
            "expression": expression,
            "sharpe": is_data.get("sharpe"),
            "turnover": is_data.get("turnover"),
            "fitness": is_data.get("fitness"),
            "returns": is_data.get("returns"),
            "drawdown": is_data.get("drawdown"),
            "margin": is_data.get("margin"),
            "long_count": is_data.get("longCount"),
            "short_count": is_data.get("shortCount"),
        }

        # Train / Test stats
        for period in ("train", "test"):
            period_data = result.get(period, {})
            if period_data:
                entry[f"{period}_sharpe"] = period_data.get("sharpe")
                entry[f"{period}_fitness"] = period_data.get("fitness")

        # Built-in checks
        for chk in checks:
            name = chk.get("name", chk.get("test", "unknown"))
            entry[f"check_{name}"] = chk.get("result", "UNKNOWN")

        # Optional checks
        if check_submission:
            try:
                sub_df = get_check_submission(s, alpha_id)
                if not sub_df.empty:
                    for _, row in sub_df.iterrows():
                        entry[f"sub_{row.get('name', row.get('test', 'unknown'))}"] = row.get("result", "UNKNOWN")
            except Exception:
                pass

        if check_self_corr:
            try:
                sc = check_self_corr_test(s, alpha_id)
                if not sc.empty:
                    entry["self_corr_result"] = sc.iloc[0].get("result")
                    entry["self_corr_value"] = sc.iloc[0].get("value")
            except Exception:
                pass

        if check_prod_corr:
            try:
                pc = check_prod_corr_test(s, alpha_id)
                if not pc.empty:
                    entry["prod_corr_result"] = pc.iloc[0].get("result")
                    entry["prod_corr_value"] = pc.iloc[0].get("value")
            except Exception:
                pass

        entry["status"] = "fetched"
        return entry

    except Exception as e:
        return {"alpha_id": alpha_id, "index": index, "status": "fetch_error", "error": str(e)}


def print_summary(submissions: list[dict]):
    """Print status summary."""
    status_counts = {}
    for s in submissions:
        st = s.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Submission Status Summary")
    print(f"{'='*60}")
    print(f"  Total entries    : {len(submissions):,}")
    for st, cnt in sorted(status_counts.items()):
        pct = cnt / len(submissions) * 100 if submissions else 0
        print(f"    {st:20s}: {cnt:>8,}  ({pct:.1f}%)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Extract results from submitted sentiment alphas")
    parser.add_argument("--poll-pending", action="store_true", help="Poll simulations still running")
    parser.add_argument("--check-submission", action="store_true", help="Run submission eligibility checks")
    parser.add_argument("--check-self-corr", action="store_true", help="Run self-correlation checks")
    parser.add_argument("--check-prod-corr", action="store_true", help="Run production-correlation checks")
    parser.add_argument("--threads", type=int, default=3, help="Concurrent result fetchers (default: 3)")
    parser.add_argument("--summary", action="store_true", help="Just print summary, no fetching")
    args = parser.parse_args()

    submissions, source_files = load_submissions()
    print_summary(submissions)

    if args.summary:
        return

    # Login
    s = start_session()

    # ── Step 1: Poll pending simulations ─────────────────────────
    if args.poll_pending:
        pending = [sub for sub in submissions if sub.get("status") in ("submitted", "interrupted")]
        if pending:
            print(f"\nPolling {len(pending):,} pending simulations...")
            resolved = 0
            for i, sub in enumerate(pending):
                poll_pending(s, sub)
                if sub["status"] not in ("submitted", "interrupted"):
                    resolved += 1
                if (i + 1) % 100 == 0:
                    print(f"  Polled {i + 1:,}/{len(pending):,}, resolved {resolved:,}")
                time.sleep(0.3)

            save_submissions(submissions)
            print(f"  Resolved {resolved:,}/{len(pending):,} pending simulations.")
        else:
            print("\nNo pending simulations to poll.")

    # ── Step 2: Fetch results for completed alphas ───────────────
    # Load existing results CSV to avoid re-fetching
    already_fetched = set()
    if os.path.exists(RESULTS_CSV):
        try:
            existing_df = pd.read_csv(RESULTS_CSV)
            already_fetched = set(existing_df["alpha_id"].dropna().astype(str))
        except Exception:
            pass

    completed = [
        sub for sub in submissions
        if sub.get("status") == "completed"
        and sub.get("alpha_id")
        and str(sub["alpha_id"]) not in already_fetched
    ]

    print(f"\nCompleted alphas   : {sum(1 for s in submissions if s.get('status') == 'completed'):,}")
    print(f"Already fetched    : {len(already_fetched):,}")
    print(f"To fetch now       : {len(completed):,}")

    if completed:
        print(f"\nFetching results with {args.threads} threads...")

        def fetch_one(sub):
            return fetch_alpha_result(
                s, sub["alpha_id"], sub.get("expression", ""), sub["index"],
                check_submission=args.check_submission,
                check_self_corr=args.check_self_corr,
                check_prod_corr=args.check_prod_corr,
            )

        new_results = []
        with ThreadPool(args.threads) as pool:
            for i, result in enumerate(pool.imap_unordered(fetch_one, completed)):
                new_results.append(result)
                if (i + 1) % 50 == 0:
                    print(f"  Fetched {i + 1:,}/{len(completed):,}")

        # Merge with existing
        if os.path.exists(RESULTS_CSV):
            try:
                existing_df = pd.read_csv(RESULTS_CSV)
                new_df = pd.DataFrame(new_results)
                df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception:
                df = pd.DataFrame(new_results)
        else:
            df = pd.DataFrame(new_results)
    else:
        if os.path.exists(RESULTS_CSV):
            df = pd.read_csv(RESULTS_CSV)
        else:
            print("\nNo results to export. Are simulations still running?")
            print("Try: python3 sentiment_alpha_results.py --poll-pending")
            return

    # ── Step 3: Save CSV ─────────────────────────────────────────
    for sort_col in ("fitness", "sharpe"):
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False, na_position="last")
            break

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults CSV saved to {RESULTS_CSV}  ({len(df):,} rows)")

    # ── Step 4: Quick summary ────────────────────────────────────
    fetched = df[df["status"] == "fetched"] if "status" in df.columns else df
    if not fetched.empty and "sharpe" in fetched.columns:
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Total fetched    : {len(fetched):,}")
        sharpe = fetched["sharpe"].dropna()
        fitness = fetched["fitness"].dropna() if "fitness" in fetched.columns else pd.Series()
        print(f"  Avg Sharpe       : {sharpe.mean():.4f}")
        print(f"  Max Sharpe       : {sharpe.max():.4f}")
        if not fitness.empty:
            print(f"  Avg Fitness      : {fitness.mean():.4f}")
            print(f"  Max Fitness      : {fitness.max():.4f}")

        # Top 10
        sort_col = "fitness" if "fitness" in fetched.columns else "sharpe"
        top = fetched.nlargest(10, sort_col)
        print(f"\n  Top 10 by {sort_col}:")
        for _, row in top.iterrows():
            print(f"    #{row.get('index','?'):>6}  sharpe={row.get('sharpe','?'):>8}  "
                  f"fitness={row.get('fitness','?'):>8}  "
                  f"id={row.get('alpha_id','?')}")
        print(f"{'='*60}")

    # Remaining
    still_pending = sum(1 for s in submissions if s.get("status") in ("submitted", "interrupted"))
    if still_pending > 0:
        print(f"\n{still_pending:,} alphas still pending. Run again with --poll-pending later.")


if __name__ == "__main__":
    main()
