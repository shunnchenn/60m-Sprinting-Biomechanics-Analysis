"""cohort.py — ETL for the whole cohort: every runner -> one feature row + one top-speed cycle.

    import cohort as C
    data = C.load()                 # cached; C.load(rebuild=True) re-reads every C3D (~1 min)
    data["features"]                # DataFrame, one row per runner
    data["cycles"][rid]             # (101, 13) mean top-speed joint-angle cycle, B-spline smoothed
    data["runs"][rid]               # sprint_summary.RunSummary (timeline, contacts, speed, ...)

Extract   sprint_summary.summarize(): events from stride.py, smoothed speed, distance.
Transform the N_STRIDES clean right-foot strides nearest top speed -> joint angles -> 101-point
          cycles -> mean -> smoothed. A stride is CLEAN when exactly one left contact sits between
          its two right touchdowns (no missed contact) and it lasts a plausible 22-40 frames.
Load      one feature row per runner, pickled to Results/cohort_events.pkl.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import sprint_pipeline as SP
import sprint_summary as SS
from project_paths import ANTHRO_PATH, PATHS

N_STRIDES = 5                     # 5 near-peak strides: best ICC in the reliability sweep
CACHE = PATHS.results / "cohort_events.pkl"


def top_speed_strides(run, n=N_STRIDES, foot="R"):
    """The `n` clean `foot` strides nearest top speed, as (touchdown, mid, next touchdown) contacts."""
    ground = [c for c in run.contacts if not c.on_blocks]
    peak = int(np.argmax(run.speed))
    clean = [(a, m, b) for a, m, b in zip(ground, ground[1:], ground[2:])
             if a.foot == b.foot == foot != m.foot
             and SP.STRIDE_FRAMES_MIN <= b.touchdown - a.touchdown <= SP.STRIDE_FRAMES_MAX]
    nearest = sorted(clean, key=lambda s: abs((s[0].touchdown + s[2].touchdown) / 2 - peak))[:n]
    return sorted(nearest, key=lambda s: s[0].touchdown)


def runner_row(run, sex):
    """Extract + transform one runner: (feature dict, smoothed mean cycle)."""
    raw, mk, fs = run.trial["raw64"], run.trial["mk"], run.trial["fs"]
    strides = top_speed_strides(run)
    windows = [(a.touchdown, b.touchdown) for a, _m, b in strides]
    cycle, n_used, spread = SP.stride_cycle_angles(raw, mk, None, windows=windows)
    cycle = SP.smooth_angle_curves(cycle)

    contacts = [c for s in strides for c in s[:2]]                        # each stride's two contacts
    nxt = {id(a): b for a, b in zip(run.contacts, run.contacts[1:])}
    row = {**run.headline, "sex": sex, "height_m": SP.PARTICIPANT_ANTHRO[run.runner]["body_height"],
           "n_strides": n_used, "phase_spread": spread,
           "top_contact_ms": np.mean([c.frames for c in contacts]) / fs * 1000,
           "top_flight_ms": np.mean([nxt[id(c)].touchdown - c.take_off - 1 for c in contacts]) / fs * 1000,
           "top_stride_length_m": np.mean([run.distance[b.touchdown] - run.distance[a.touchdown]
                                           for a, _m, b in strides]),
           "top_stride_frequency_hz": np.mean([fs / (b.touchdown - a.touchdown) for a, _m, b in strides]),
           **SP.scalar_features(cycle)}
    return row, cycle


def build():
    sex = (pd.read_excel(ANTHRO_PATH).assign(rid=lambda d: d["Subject Number"].astype(str).str.split("-").str[0])
           .set_index("rid")["Sex"].str.strip().str.upper().str[0])
    rows, cycles, runs = [], {}, {}
    for f in sorted(SP.C3D_DIR.glob("*.c3d")):
        rid = f.stem.split("-")[0].strip()
        if rid in SP.EXCLUDED_PIDS:
            continue
        runs[rid] = SS.summarize(rid)
        row, cycles[rid] = runner_row(runs[rid], sex[rid])
        rows.append(row)
    data = {"features": pd.DataFrame(rows).set_index("runner"), "cycles": cycles, "runs": runs}
    PATHS.results.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(data, fh)
    return data


def load(rebuild=False):
    if rebuild or not CACHE.exists():
        return build()
    with open(CACHE, "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    d = build()
    feat = d["features"]
    assert len(feat) == 30 and (feat.n_strides >= 3).all(), feat.n_strides.describe()
    assert abs(feat.loc["SB80", "top_speed_ms"] - 7.14) < 0.005
    print(f"wrote {CACHE.name}: {feat.shape[0]} runners x {feat.shape[1]} columns")
    print(feat[["sex", "top_speed_ms", "n_strides", "phase_spread", "top_contact_ms", "top_flight_ms"]]
          .describe().round(2))
