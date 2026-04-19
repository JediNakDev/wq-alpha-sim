# Alpha Creation Engine

Automated alpha generation, simulation, and submission pipeline for the [WorldQuant BRAIN](https://platform.worldquantbrain.com) platform.

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Project layout

```
ace_lib.py                  # Core API library (session, simulation, submission)
helpful_functions.py        # Utility helpers used by ace_lib
alpha_pipeline.py           # Shared simulate / tag / submit engine
alpha_choices.py            # Interactive region / universe / delay picker (API-driven)

sentiment_alpha_generator.py    # Generate sentiment alphas
sentiment_alpha_simulator.py    # Simulate + tag + submit sentiment alphas

fundamental_alpha_generator.py  # Generate fundamental ratio alphas
fundamental_alpha_simulator.py  # Simulate + tag + submit fundamental alphas

residual_alpha_generator.py     # Generate sector/size-neutralized residual alphas

template/
  how_to_use.ipynb          # Interactive API walkthrough

alphas/                     # Generated JSONL alpha files (gitignored)
  sentiment_<region>_<universe>/
  fundamental_<region>_<universe>/
  residual_<region>_<universe>/

results/                    # Simulation tracking files (gitignored)
  <domain>_<region>_<universe>_d<delay>/
    sub_<batch>.jsonl       # Per-batch simulation tracking
    tag_submit_done.jsonl   # Tag + submit log
```

## Alpha domains

| Domain | Template | Generator | Simulator |
|--------|----------|-----------|-----------|
| Sentiment | `ts_op(compare(rank(backfill(pos)), rank(backfill(neg))), days)` | `sentiment_alpha_generator.py` | `sentiment_alpha_simulator.py` |
| Fundamental | `ts_op(group_rank(profit) / (group_rank(size) + 1), days)` | `fundamental_alpha_generator.py` | `fundamental_alpha_simulator.py` |
| Residual | `ts_regression(winsorize(ts_backfill(data, 63)), group_mean, 252)` | `residual_alpha_generator.py` | *(use fundamental simulator)* |

## Usage

### 1. Generate alphas

Running any generator without flags launches an **interactive picker** that fetches valid region / universe / delay options live from the API:

```bash
python3 sentiment_alpha_generator.py        # interactive
python3 sentiment_alpha_generator.py --region USA --universe TOP3000 --delay 1  # scripted

python3 fundamental_alpha_generator.py      # interactive (default: HKG / TOP500 / delay 1)
python3 residual_alpha_generator.py         # interactive
```

Output lands in `alphas/<domain>_<region>_<universe>/`.

### 2. Simulate alphas

```bash
python3 sentiment_alpha_simulator.py --input alphas/sentiment_usa_top3000/d005_subtract.jsonl
python3 fundamental_alpha_simulator.py --input alphas/fundamental_hkg_top500/d005_ts_delta.jsonl

# Resume (auto-skips already completed indices)
python3 sentiment_alpha_simulator.py --input alphas/sentiment_usa_top3000/d005_subtract.jsonl

# Start fresh, ignore prior progress
python3 sentiment_alpha_simulator.py --input alphas/sentiment_usa_top3000/d005_subtract.jsonl --fresh
```

Simulators also prompt for region / universe / delay if not provided via flags, or accept `--region`, `--universe`, `--delay`.

Key flags (both simulators):

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Input JSONL file |
| `--region/--universe/--delay` | interactive | Simulation target |
| `--threads` | `8` | Parallel sim threads |
| `--batch-size` | `10` | Alphas per multi-sim batch |
| `--retry-wait` | `60` | Base seconds to wait on 429 |
| `--fresh` | off | Wipe tracking and restart |
| `--start-index` | auto | Override starting alpha index |

Tracking files are written to `results/<domain>_<region>_<universe>_d<delay>/sub_<batch>.jsonl`. Qualified alphas (sharpe ≥ 1.0, turnover ≤ 70 %) are automatically tagged, described, and submitted.

## Tutorial

```bash
jupyter notebook template/how_to_use.ipynb
```
