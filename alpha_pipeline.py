"""
Alpha Pipeline (shared simulate / tag / submit engine)
======================================================
Domain-agnostic simulation pipeline used by sentiment, fundamental, and
residual alpha generators. A single entrypoint, `simulate_alphas`, accepts
an alpha JSONL input file, a settings dict, a description string and a
category — then runs the two-engine (sim + submit) workflow.

A small helper, `write_alphas_jsonl`, is used by generators to serialize
expressions into JSONL files consumed by the simulator.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Iterable

from ace_lib import (
    brain_api_url,
    check_session_and_relogin,
    get_simulation_result_json,
    set_alpha_properties,
    start_session,
    start_simulation,
    submit_alpha,
    SingleSession,
)


# ── Settings helper ───────────────────────────────────────────

def build_settings(
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "SLOW_AND_FAST",
    truncation: float = 0.08,
    pasteurization: str = "ON",
    unit_handling: str = "VERIFY",
    nan_handling: str = "OFF",
    language: str = "FASTEXPR",
    instrument_type: str = "EQUITY",
    visualization: bool = False,
) -> dict:
    return {
        "instrumentType": instrument_type,
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
        "pasteurization": pasteurization,
        "unitHandling": unit_handling,
        "nanHandling": nan_handling,
        "language": language,
        "visualization": visualization,
    }


def results_dir_for(domain: str, settings: dict) -> str:
    region = settings.get("region", "USA").lower()
    universe = settings.get("universe", "TOP3000").lower()
    delay = settings.get("delay", 1)
    return os.path.join("results", f"{domain}_{region}_{universe}_d{delay}")


def alphas_dir_for(domain: str, settings: dict) -> str:
    region = settings.get("region", "USA").lower()
    universe = settings.get("universe", "TOP3000").lower()
    return os.path.join("alphas", f"{domain}_{region}_{universe}")


def infer_settings_from_path(input_file: str, delay: int = 1, **overrides) -> dict:
    """Infer region + universe from an alphas/<domain>_<region>_<universe>/... path.

    E.g. alphas/sentiment_usa_top3000/d005_subtract.jsonl → region=USA, universe=TOP3000
    All other settings use build_settings defaults unless supplied via overrides.
    """
    import re
    dirname = os.path.basename(os.path.dirname(os.path.abspath(input_file)))
    m = re.match(r"^[a-z]+_([a-z]+)_([a-z0-9]+)$", dirname)
    if not m:
        print(f"ERROR: Cannot infer region/universe from path '{dirname}'.")
        print("  Expected format: alphas/<domain>_<region>_<universe>/<file>.jsonl")
        sys.exit(1)
    region = m.group(1).upper()
    universe = m.group(2).upper()
    return build_settings(region=region, universe=universe, delay=delay, **overrides)


# ── Generator helper ──────────────────────────────────────────

def write_alphas_jsonl(expressions: Iterable[str], output_path: str) -> int:
    """Write regular alpha expressions as JSONL ({type, regular} per line).

    The simulator fills in `settings` at simulation time, so the file is
    simulation-config-agnostic and reusable across runs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for expr in expressions:
            f.write(json.dumps({"type": "REGULAR", "regular": expr}) + "\n")
            count += 1
    return count


# ── File helpers ──────────────────────────────────────────────

_file_lock = threading.Lock()


def append_record(filepath: str, entry: dict):
    with _file_lock:
        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")


def get_submissions_file(input_file: str, results_dir: str) -> str:
    base = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(results_dir, f"sub_{base}.jsonl")


def load_alphas(input_file: str, settings: dict) -> list[dict]:
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    alphas = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            alpha = json.loads(line)
            alpha["settings"] = dict(settings)
            alpha.setdefault("type", "REGULAR")
            alphas.append(alpha)
    print(f"Loaded {len(alphas):,} alphas from {input_file}")
    return alphas


def load_done_indices(submissions_file: str) -> set[int]:
    """Only 'completed' counts as done; sim_failed will be retried."""
    indices = set()
    if not os.path.exists(submissions_file):
        return indices
    with open(submissions_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("status") == "completed":
                    indices.add(entry["index"])
            except json.JSONDecodeError:
                continue
    return indices


def clean_failed_from_file(submissions_file: str) -> int:
    if not os.path.exists(submissions_file):
        return 0
    kept, removed = [], 0
    with open(submissions_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("status") == "sim_failed":
                    removed += 1
                    continue
            except json.JSONDecodeError:
                pass
            kept.append(line)
    if removed > 0:
        with open(submissions_file, "w") as f:
            for line in kept:
                f.write(line + "\n")
    return removed


# ── Sim Engine ────────────────────────────────────────────────

def simulate_batch(
    s: SingleSession,
    batch: list[tuple[int, dict]],
    retry_wait: int,
) -> list[dict]:
    """POST a multi-simulation batch, poll to completion, return results.
    Retries on 429 with escalating backoff."""
    indices = [idx for idx, _ in batch]
    alpha_datas = [data for _, data in batch]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            s = check_session_and_relogin(s)

            payload = alpha_datas[0] if len(alpha_datas) == 1 else alpha_datas
            sim_resp = start_simulation(s, payload)

            if sim_resp.status_code == 429:
                wait = retry_wait * (attempt + 1)
                print(f"    ⏸  429 on batch (idx {indices[0]}–{indices[-1]}), "
                      f"waiting {wait}s (attempt {attempt+1}/{max_retries})…")
                time.sleep(wait)
                continue

            if sim_resp.status_code // 100 != 2:
                error = sim_resp.text[:300]
                return [
                    {"index": idx, "status": "sim_failed", "alpha_id": None,
                     "expression": d.get("regular", ""),
                     "error": f"HTTP {sim_resp.status_code}: {error}"}
                    for idx, d in batch
                ]

            progress_url = sim_resp.headers.get("Location", "")
            if not progress_url:
                return [
                    {"index": idx, "status": "sim_failed", "alpha_id": None,
                     "expression": d.get("regular", ""), "error": "No Location header"}
                    for idx, d in batch
                ]

            while True:
                prog_resp = s.get(progress_url)
                if prog_resp.status_code == 429:
                    time.sleep(retry_wait)
                    continue
                if prog_resp.status_code // 100 != 2:
                    time.sleep(10)
                    continue
                retry_after = prog_resp.headers.get("Retry-After", 0)
                if retry_after != 0 and str(retry_after) != "0":
                    time.sleep(float(retry_after))
                    continue
                break

            prog_data = prog_resp.json()

            if len(alpha_datas) == 1:
                alpha_id = prog_data.get("alpha")
                if prog_data.get("status") == "ERROR" or not alpha_id:
                    return [{"index": indices[0], "status": "sim_failed", "alpha_id": None,
                             "expression": alpha_datas[0].get("regular", ""),
                             "error": json.dumps(prog_data)[:300]}]
                return [{"index": indices[0], "status": "completed",
                         "alpha_id": alpha_id,
                         "expression": alpha_datas[0].get("regular", "")}]

            children = prog_data.get("children", [])
            if not children:
                return [
                    {"index": idx, "status": "sim_failed", "alpha_id": None,
                     "expression": d.get("regular", ""), "error": "No children"}
                    for idx, d in batch
                ]

            out = []
            for i, child_id in enumerate(children):
                idx = indices[i] if i < len(indices) else -1
                expr = alpha_datas[i].get("regular", "") if i < len(alpha_datas) else ""
                try:
                    child_resp = s.get(brain_api_url + "/simulations/" + child_id)
                    alpha_id = child_resp.json().get("alpha")
                    if alpha_id:
                        out.append({"index": idx, "status": "completed",
                                    "alpha_id": alpha_id, "expression": expr})
                    else:
                        out.append({"index": idx, "status": "sim_failed", "alpha_id": None,
                                    "expression": expr, "error": f"Child {child_id} no alpha"})
                except Exception as e:
                    out.append({"index": idx, "status": "sim_failed", "alpha_id": None,
                                "expression": expr, "error": str(e)})
            return out

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    ⚠  Batch error (attempt {attempt+1}): {e}, retrying in {retry_wait}s…")
                time.sleep(retry_wait)
                continue
            return [
                {"index": idx, "status": "sim_failed", "alpha_id": None,
                 "expression": d.get("regular", ""), "error": str(e)}
                for idx, d in batch
            ]

    return [
        {"index": idx, "status": "sim_failed", "alpha_id": None,
         "expression": d.get("regular", ""), "error": "429 retries exhausted"}
        for idx, d in batch
    ]


# ── Submit Engine ─────────────────────────────────────────────

class SubmitEngine(threading.Thread):
    """Drains a queue of completed alphas, fetches results, tags + submits
    qualified ones. Pauses on 429."""

    def __init__(
        self,
        s: SingleSession,
        submit_queue: queue.Queue,
        submissions_file: str,
        tag_submit_file: str,
        retry_wait: int,
        description: str,
        category: str,
        tag: str,
        min_sharpe: float,
        max_turnover: float,
        counters: dict,
        counters_lock: threading.Lock,
    ):
        super().__init__(daemon=True)
        self.s = s
        self.q = submit_queue
        self.submissions_file = submissions_file
        self.tag_submit_file = tag_submit_file
        self.retry_wait = retry_wait
        self.description = description
        self.category = category
        self.tag = tag
        self.min_sharpe = min_sharpe
        self.max_turnover = max_turnover
        self.counters = counters
        self.lock = counters_lock
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set() or not self.q.empty():
            try:
                item = self.q.get(timeout=2)
            except queue.Empty:
                continue

            alpha_id = item["alpha_id"]
            index = item["index"]
            expression = item.get("expression", "")

            record = self._process_one(alpha_id, index, expression)
            append_record(self.submissions_file, record)

            ts_record = {
                "alpha_id": alpha_id,
                "sharpe": record.get("sharpe"),
                "turnover": record.get("turnover"),
                "fitness": record.get("fitness"),
            }
            if record.get("qualified"):
                ts_record["action"] = "tagged_and_submitted"
                ts_record["tag_status"] = record.get("tag_status")
                ts_record["submit_status"] = record.get("submit_status")
            else:
                ts_record["action"] = "not_qualified"
            append_record(self.tag_submit_file, ts_record)

            if record.get("qualified"):
                with self.lock:
                    self.counters["qualified"] += 1
                print(
                    f"    ★ QUALIFIED {alpha_id}  "
                    f"sharpe={record.get('sharpe')}  "
                    f"turnover={record.get('turnover')}  "
                    f"submit={record.get('submit_status')}  "
                    f"→ tagged + described + submitted"
                )

            self.q.task_done()

    def _process_one(self, alpha_id: str, index: int, expression: str) -> dict:
        record = {
            "index": index,
            "status": "completed",
            "alpha_id": alpha_id,
            "expression": expression,
            "completed_at": datetime.now().isoformat(),
        }

        try:
            result = self._call_with_429_retry(
                lambda: get_simulation_result_json(self.s, alpha_id)
            )
            if not result:
                return record

            is_data = result.get("is", {})
            sharpe = is_data.get("sharpe")
            turnover = is_data.get("turnover")
            fitness = is_data.get("fitness")

            record["sharpe"] = sharpe
            record["turnover"] = turnover
            record["fitness"] = fitness

            if (sharpe is not None and turnover is not None
                    and sharpe >= self.min_sharpe and turnover <= self.max_turnover):
                record["qualified"] = True

                resp = self._call_with_429_retry(
                    lambda: set_alpha_properties(self.s, alpha_id, tags=[self.tag])
                )
                record["tag_status"] = resp.status_code if resp else None

                resp = self._call_with_429_retry(
                    lambda: set_alpha_properties(self.s, alpha_id, regular_desc=self.description)
                )
                record["desc_status"] = resp.status_code if resp else None

                resp = self._call_with_429_retry(
                    lambda: set_alpha_properties(self.s, alpha_id, category=self.category)
                )
                record["cat_status"] = resp.status_code if resp else None

                resp = self._call_with_429_retry(
                    lambda: submit_alpha(self.s, alpha_id)
                )
                record["submit_status"] = resp.status_code if resp else None
            else:
                record["qualified"] = False

        except Exception as e:
            record["result_error"] = str(e)

        return record

    def _call_with_429_retry(self, func, max_retries: int = 10):
        result = None
        for attempt in range(max_retries):
            self.s = check_session_and_relogin(self.s)
            result = func()
            if hasattr(result, "status_code") and result.status_code == 429:
                wait = self.retry_wait * (attempt + 1)
                print(f"      ⏸  Submit engine 429, waiting {wait}s "
                      f"(attempt {attempt+1}/{max_retries})…")
                time.sleep(wait)
                continue
            return result
        return result


# ── Pipeline orchestration ────────────────────────────────────

def _run_pipeline(
    s: SingleSession,
    alpha_list: list[dict],
    tracking_file: str,
    tag_submit_file: str,
    description: str,
    category: str,
    tag: str,
    min_sharpe: float,
    max_turnover: float,
    threads: int,
    batch_size: int,
    retry_wait: int,
    label: str,
) -> dict:
    total = len(alpha_list)
    already_done = load_done_indices(tracking_file)
    work = [(i, alpha_list[i]) for i in range(total) if i not in already_done]

    if not work:
        print(f"  [{label}] all {total:,} alphas already done.")
        return {"completed": 0, "failed": 0, "qualified": 0}

    batches = [work[i:i + batch_size] for i in range(0, len(work), batch_size)]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total alphas   : {total:,}")
    print(f"  Already done   : {len(already_done):,}")
    print(f"  To simulate    : {len(work):,}")
    print(f"  Batches        : {len(batches):,}  ({batch_size} per batch)")
    print(f"  Tracking file  : {tracking_file}")
    print(f"{'='*60}\n")

    completed_count = 0
    failed_count = 0
    counters = {"qualified": 0}
    counters_lock = threading.Lock()

    submit_q: queue.Queue = queue.Queue()
    submit_engine = SubmitEngine(
        s, submit_q, tracking_file, tag_submit_file, retry_wait,
        description, category, tag, min_sharpe, max_turnover,
        counters, counters_lock,
    )
    submit_engine.start()
    print(f"  [{label}] Submit engine started\n")

    sim_func = partial(simulate_batch, s, retry_wait=retry_wait)

    try:
        with ThreadPool(threads) as pool:
            for batch_i, batch_results in enumerate(
                pool.imap_unordered(sim_func, batches)
            ):
                for res in batch_results:
                    if res["status"] == "completed" and res.get("alpha_id"):
                        completed_count += 1
                        submit_q.put(res)
                    else:
                        failed_count += 1
                        append_record(tracking_file, res)

                done_total = completed_count + failed_count
                pct = (done_total / len(work) * 100) if work else 0
                q_count = counters["qualified"]
                pending = submit_q.qsize()
                print(
                    f"  [{label}] [{pct:5.1f}%] batch {batch_i+1}/{len(batches)}  "
                    f"simulated={completed_count}  failed={failed_count}  "
                    f"qualified={q_count}  submit_pending={pending}"
                )
    except KeyboardInterrupt:
        print(f"\n\n[{label}] Interrupted! Waiting for submit engine to drain…")

    remaining = submit_q.qsize()
    if remaining > 0:
        print(f"\n  [{label}] Sim engine done. Waiting for submit engine "
              f"to process {remaining} remaining…")

    submit_q.join()
    submit_engine.stop()
    submit_engine.join(timeout=30)

    q_count = counters["qualified"]
    print(f"\n  [{label}] DONE — simulated={completed_count}  "
          f"qualified={q_count}  failed={failed_count}")

    return {"completed": completed_count, "failed": failed_count, "qualified": q_count}


def simulate_alphas(
    input_file: str,
    settings: dict,
    description: str,
    category: str,
    results_dir: str,
    tag: str = "PowerPoolSelected",
    min_sharpe: float = 1.0,
    max_turnover: float = 0.70,
    threads: int = 8,
    batch_size: int = 10,
    retry_wait: int = 60,
    fresh: bool = False,
    start_index: int | None = None,
):
    """Simulate alphas from a JSONL file, tag + describe + submit qualified.

    Args:
        input_file: Path to JSONL of {type, regular} alpha entries.
        settings: Simulation settings dict (see `build_settings`).
        description: Long-form rationale written onto qualified alphas.
        category: Category string applied to qualified alphas (e.g. "SENTIMENT").
        results_dir: Directory for tracking files.
        tag: Tag string applied to qualified alphas.
        min_sharpe / max_turnover: Qualification thresholds.
        threads / batch_size: Concurrency controls for the sim engine.
        retry_wait: Base seconds to wait on 429.
        fresh: Wipe tracking file and resimulate from scratch.
        start_index: Override starting alpha index.
    """
    threads = min(max(threads, 1), 8)
    batch_size = min(max(batch_size, 1), 10)

    os.makedirs(results_dir, exist_ok=True)
    submissions_file = get_submissions_file(input_file, results_dir)
    tag_submit_file = os.path.join(results_dir, "tag_submit_done.jsonl")

    if fresh and os.path.exists(submissions_file):
        os.remove(submissions_file)
        print(f"  Wiped {submissions_file} (--fresh)")
    if not fresh:
        removed = clean_failed_from_file(submissions_file)
        if removed > 0:
            print(f"  Cleaned {removed:,} sim_failed entries for retry")

    print(f"\n{'='*60}")
    print(f"  Alpha Simulator ({category})")
    print(f"{'='*60}")
    print(f"  Region           : {settings.get('region')}")
    print(f"  Universe         : {settings.get('universe')}")
    print(f"  Delay            : {settings.get('delay')}")
    print(f"  Neutralization   : {settings.get('neutralization')}")
    print(f"  Truncation       : {settings.get('truncation')}")
    print(f"  Input file       : {input_file}")
    print(f"  Results dir      : {results_dir}")
    print(f"  Sim threads      : {threads}")
    print(f"  Batch size       : {batch_size}")
    print(f"  429 wait time    : {retry_wait}s")
    print(f"  Qualification    : sharpe >= {min_sharpe}, turnover <= {max_turnover}")
    print(f"{'='*60}")

    s = start_session()

    alpha_list = load_alphas(input_file, settings)
    if start_index is not None:
        alpha_list = alpha_list[start_index:]

    summary = _run_pipeline(
        s, alpha_list, submissions_file, tag_submit_file,
        description, category, tag, min_sharpe, max_turnover,
        threads, batch_size, retry_wait,
        label=os.path.basename(input_file),
    )

    print(f"\n{'='*60}")
    print(f"  SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Simulated  : {summary['completed']:,}")
    print(f"  Qualified  : {summary['qualified']:,}  "
          f"(sharpe >= {min_sharpe} & turnover <= {max_turnover})")
    print(f"  Sim failed : {summary['failed']:,}")
    print(f"  Saved to   : {submissions_file}")
    print(f"{'='*60}")
