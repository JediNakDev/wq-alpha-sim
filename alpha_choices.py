"""
Alpha Configuration Picker
==========================
Interactive region / universe / delay chooser backed by live API data.

Usage (from a generator script):
    from alpha_choices import pick_region_universe_delay
    s = start_session()
    region, universe, delay = pick_region_universe_delay(
        s, default_region="HKG", default_universe="TOP500", default_delay=1
    )
"""

from __future__ import annotations

import pandas as pd

from ace_lib import get_instrument_type_region_delay, SingleSession

_cached_df: pd.DataFrame | None = None


def _get_options(s: SingleSession, instrument_type: str = "EQUITY") -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        _cached_df = get_instrument_type_region_delay(s)
    return _cached_df[_cached_df["InstrumentType"] == instrument_type].copy()


def _menu(prompt: str, options: list, default: str | int | None) -> str | int:
    """Print a numbered menu and return the chosen value."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if str(opt) == str(default) else ""
        print(f"  {i}) {opt}{marker}")

    while True:
        raw = input("  Enter number (or press Enter for default): ").strip()
        if raw == "":
            if default is not None and default in options:
                return default
            if default is not None:
                try:
                    return options[int(default) - 1]
                except (ValueError, IndexError):
                    pass
            return options[0]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"  Invalid — enter a number between 1 and {len(options)}.")


def pick_region_universe_delay(
    s: SingleSession,
    default_region: str | None = None,
    default_universe: str | None = None,
    default_delay: int | None = None,
    instrument_type: str = "EQUITY",
) -> tuple[str, str, int]:
    """Display step-by-step menus for region → delay → universe.

    Menus are built from live API data so they always reflect what the
    platform currently supports.  Passing CLI-resolved defaults pre-selects
    the appropriate option; hitting Enter accepts it.

    Returns:
        (region, universe, delay)
    """
    df = _get_options(s, instrument_type)

    print(f"\n{'='*50}")
    print(f"  Configure simulation target")
    print(f"{'='*50}")

    # ── Step 1: Region ────────────────────────────────────────
    regions = sorted(df["Region"].unique())
    region = _menu("Step 1 — Select region:", regions, default_region)

    # ── Step 2: Delay ─────────────────────────────────────────
    region_df = df[df["Region"] == region]
    delays = sorted(region_df["Delay"].unique())
    delay = _menu("Step 2 — Select delay:", delays, default_delay)
    delay = int(delay)

    # ── Step 3: Universe ──────────────────────────────────────
    row = region_df[region_df["Delay"] == delay].iloc[0]
    universes = row["Universe"]
    universe = _menu("Step 3 — Select universe:", universes, default_universe)

    print(f"\n  Target: {instrument_type} | {region} | {universe} | delay={delay}")
    print(f"{'='*50}\n")

    return str(region), str(universe), int(delay)
