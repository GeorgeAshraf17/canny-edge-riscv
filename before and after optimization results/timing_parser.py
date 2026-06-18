# tools/python/timing_parser.py
# ---------------------------------------------------------------------------
# Shared helpers for the docs/*.txt timing files used by tools/python/plot*.py
#
# This repo's pipeline (see src/main.cpp) only times four stages:
#   Gaussian -> Sobel -> Magnitude -> Direction
# There is no NMS / double-threshold / hysteresis stage implemented anywhere
# in src/, so STAGES intentionally stops at "Direction". Add to this list
# (and to main.cpp's timing block) if those stages get implemented later.
# ---------------------------------------------------------------------------
from __future__ import annotations
import os

STAGES = ["Gaussian", "Sobel", "Magnitude", "Direction"]


def _read_rows(path: str):
    """Yield (tokens) for each non-blank, non-comment line in a timing file."""
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line.split()


def parse_timing_file(path: str, col: int = 0) -> dict[str, float] | None:
    """
    Parse a simple "padded" timing file:
        <Stage> <value_col0> [<value_col1> ...]

    `col` selects which value column to use (0-indexed, after the stage
    name). Returns {stage: value_us}, or None if the file doesn't exist.
    """
    rows = list(_read_rows(path))
    if not rows:
        return None

    out: dict[str, float] = {}
    for tokens in rows:
        stage = tokens[0]
        if stage not in STAGES:
            continue
        values = tokens[1:]
        if col >= len(values):
            continue
        try:
            out[stage] = float(values[col])
        except ValueError:
            continue
    return out or None


def parse_speedup_file(path: str):
    """
    Parse a "speedup target" file with one row per stage:
        <Stage> <scalar_us> <rvv_us> <vectorized: yes|no>

    The 4th column records whether that stage actually runs RVV intrinsics
    in the *default* canny_rvv build (see Makefile's canny_rvv recipe and
    the #ifdef USE_RVV_SOBEL / USE_RVV_GAUSSIAN guards in src/main.cpp).
    This matters because it's easy to *assume* a stage is vectorized just
    because an *_rvv.cpp file exists for it (e.g. sobel_rvv.cpp exists and
    is compiled into canny_rvv, but USE_RVV_SOBEL is never defined, so it's
    never actually called at runtime -- the scalar sobel() runs instead).

    If rvv_us is missing/blank for a non-vectorized stage, the scalar value
    is reused automatically (true scalar-fallback timing), rather than
    trusting a possibly-stale recorded number.

    Returns (scalar_dict, rvv_dict, vectorized_set) or (None, None, None)
    if the file is missing.
    """
    rows = list(_read_rows(path))
    if not rows:
        return None, None, None

    scalar: dict[str, float] = {}
    rvv: dict[str, float] = {}
    vectorized: set[str] = set()

    for tokens in rows:
        if len(tokens) < 3:
            continue
        stage = tokens[0]
        if stage not in STAGES:
            continue
        try:
            sc_us = float(tokens[1])
        except ValueError:
            continue

        rvv_raw = tokens[2]
        is_vectorized = len(tokens) >= 4 and tokens[3].lower() in ("yes", "true", "1")

        try:
            rv_us = float(rvv_raw)
        except ValueError:
            rv_us = sc_us  # no recorded value -> fall back to scalar time

        # Belt-and-suspenders: a stage marked "not vectorized" always gets
        # the scalar fallback time, regardless of whatever number is in
        # column 3 -- this is what guards against stale/inconsistent data
        # in the source file silently mislabeling a scalar stage as fast.
        if not is_vectorized:
            rv_us = sc_us
        else:
            vectorized.add(stage)

        scalar[stage] = sc_us
        rvv[stage] = rv_us

    if not scalar:
        return None, None, None
    return scalar, rvv, vectorized
