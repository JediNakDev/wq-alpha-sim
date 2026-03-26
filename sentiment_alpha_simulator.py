"""
Program 2: Sentiment Alpha Submitter (Slot-Managed)
=====================================================
Reads a generated_alphas/*.jsonl file and submits simulations while
respecting the platform's concurrent simulation limit (~8).

Each input file gets its own tracking file, e.g.:
    input:  generated_alphas/d005_subtract.jsonl
    output: sentiment_results/sub_d005_subtract.jsonl

Usage:
    python3 sentiment_alpha_simulator.py --input generated_alphas/d005_subtract.jsonl
    python3 sentiment_alpha_simulator.py --input generated_alphas/d020_divide.jsonl --max-concurrent 6
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from ace_lib import (
    brain_api_url,
    check_session_and_relogin,
    start_session,
    start_simulation,
    SingleSession,
)

RESULTS_DIR = "sentiment_results"


def get_submissions_file(input_file: str) -> str:
    """Derive submissions filename from input filename."""
    base = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(RESULTS_DIR, f"sub_{base}.jsonl")


def load_alphas(input_file: str) -> list[dict]:
    """Load alpha list from JSONL file."""
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Run sentiment_alpha_generator.py first.")
        sys.exit(1)

    alphas = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                alphas.append(json.loads(line))

    print(f"Loaded {len(alphas):,} alphas from {input_file}")
    return alphas


def load_submitted_indices(submissions_file: str) -> set[int]:
    """Load already-submitted indices from submissions file."""
    indices = set()
    if os.path.exists(submissions_file):
        with open(submissions_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get("status") in ("submitted", "completed", "sim_failed"):
                            indices.add(entry["index"])
                    except json.JSONDecodeError:
                        continue
    return indices


def append_submission(submissions_file: str, entry: dict):
    """Append a single submission record to the JSONL file."""
    with open(submissions_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def submit_one(s: SingleSession, alpha_data: dict, index: int) -> dict:
    """Submit a single alpha, return tracking dict with progress_url."""
    try:
        response = start_simulation(s, alpha_data)

        if response.status_code // 100 != 2:
            detail = response.text[:500]
            return {
                "index": index,
                "status": "submit_failed",
                "error": detail,
                "expression": alpha_data.get("regular", ""),
                "progress_url": None,
                "alpha_id": None,
                "submitted_at": datetime.now().isoformat(),
            }

        progress_url = response.headers.get("Location", "")
        return {
            "index": index,
            "status": "submitted",
            "expression": alpha_data.get("regular", ""),
            "progress_url": progress_url,
            "alpha_id": None,
            "submitted_at": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "index": index,
            "status": "submit_error",
            "error": str(e),
            "expression": alpha_data.get("regular", ""),
            "progress_url": None,
            "alpha_id": None,
            "submitted_at": datetime.now().isoformat(),
        }


def check_slot(s: SingleSession, slot: dict) -> dict:
    """Poll a submitted simulation's progress URL."""
    progress_url = slot.get("progress_url")
    if not progress_url:
        return slot

    try:
        resp = s.get(progress_url)

        if resp.status_code // 100 != 2:
            return slot

        retry_after = resp.headers.get("Retry-After", 0)
        if retry_after != 0 and str(retry_after) != "0":
            return slot  # still running

        data = resp.json()
        status = data.get("status", "UNKNOWN")
        alpha_id = data.get("alpha", None)

        if status == "ERROR" or alpha_id is None:
            slot["status"] = "sim_failed"
            slot["error"] = json.dumps(data)[:500]
            slot["alpha_id"] = None
        else:
            slot["status"] = "completed"
            slot["alpha_id"] = alpha_id

        slot["completed_at"] = datetime.now().isoformat()
        return slot

    except Exception:
        return slot


def main():
    parser = argparse.ArgumentParser(description="Submit sentiment alphas with slot management")
    parser.add_argument("--input", required=True, help="Input JSONL file (e.g. generated_alphas/d005_subtract.jsonl)")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Max concurrent simulations (default: 8)")
    parser.add_argument("--start-index", type=int, default=None, help="Start from this alpha index")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="Seconds between polling active slots (default: 3)")
    parser.add_argument("--session-check", type=int, default=500, help="Re-check session every N submissions (default: 500)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    submissions_file = get_submissions_file(args.input)

    # Load alphas
    alpha_list = load_alphas(args.input)
    total = len(alpha_list)

    # Load already-submitted indices
    already_done = load_submitted_indices(submissions_file)

    # Determine starting point
    if args.start_index is not None:
        start = args.start_index
    elif already_done:
        start = max(already_done) + 1
    else:
        start = 0

    if start >= total:
        print(f"All {total:,} alphas already submitted.")
        print(f"Run sentiment_alpha_results.py to fetch results.")
        return

    remaining = total - start

    print(f"\n{'='*60}")
    print(f"  Sentiment Alpha Submitter (Slot-Managed)")
    print(f"{'='*60}")
    print(f"  Input file       : {args.input}")
    print(f"  Total alphas     : {total:,}")
    print(f"  Already done     : {len(already_done):,}")
    print(f"  Starting from    : #{start:,}")
    print(f"  Remaining        : {remaining:,}")
    print(f"  Max concurrent   : {args.max_concurrent}")
    print(f"  Poll interval    : {args.poll_interval}s")
    print(f"  Tracking file    : {submissions_file}")
    print(f"{'='*60}\n")

    # Login
    s = start_session()

    active_slots: list[dict] = []
    next_index = start
    submitted_count = 0
    completed_count = 0
    failed_count = 0
    total_session_submissions = 0

    try:
        while next_index < total or len(active_slots) > 0:

            # ── 1. Fill empty slots with new submissions ─────────────
            while len(active_slots) < args.max_concurrent and next_index < total:
                if next_index in already_done:
                    next_index += 1
                    continue

                total_session_submissions += 1
                if total_session_submissions % args.session_check == 0:
                    print(f"\n  -- Re-checking session after {total_session_submissions} submissions --")
                    s = check_session_and_relogin(s)

                result = submit_one(s, alpha_list[next_index], next_index)

                if result["status"] == "submitted":
                    active_slots.append(result)
                    submitted_count += 1
                else:
                    append_submission(submissions_file, result)
                    failed_count += 1

                next_index += 1

            # ── 2. Poll active slots for completions ─────────────────
            if not active_slots:
                break

            time.sleep(args.poll_interval)

            still_active = []
            batch_done = 0

            for slot in active_slots:
                updated = check_slot(s, slot)
                if updated["status"] in ("completed", "sim_failed"):
                    append_submission(submissions_file, updated)
                    if updated["status"] == "completed":
                        completed_count += 1
                    else:
                        failed_count += 1
                    batch_done += 1
                else:
                    still_active.append(updated)

            active_slots = still_active

            # ── 3. Progress display ──────────────────────────────────
            if batch_done > 0:
                done_total = completed_count + failed_count
                pct = (done_total / remaining * 100) if remaining > 0 else 0
                print(
                    f"  [{pct:5.1f}%] submitted={submitted_count}  completed={completed_count}  "
                    f"failed={failed_count}  active={len(active_slots)}  "
                    f"next=#{next_index:,}/{total:,}"
                )

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving {len(active_slots)} active slots...")
        for slot in active_slots:
            slot["status"] = "interrupted"
            append_submission(submissions_file, slot)

    print(f"\n{'='*60}")
    print(f"  SESSION SUMMARY")
    print(f"  Submitted  : {submitted_count:,}")
    print(f"  Completed  : {completed_count:,}")
    print(f"  Failed     : {failed_count:,}")
    print(f"  Saved to   : {submissions_file}")
    print(f"{'='*60}")
    print(f"\nNext: python3 sentiment_alpha_results.py")


if __name__ == "__main__":
    main()
