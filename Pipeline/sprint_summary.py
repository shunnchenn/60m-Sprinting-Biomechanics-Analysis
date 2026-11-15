"""sprint_summary.py — the 60 m run in plain numbers, before any kinematics.

    import sprint_summary as SS
    run = SS.summarize("SB80")
    run.timeline        # every event in the order it happens
    run.headline        # the numbers most people ask for first

Each block below computes ONE moment of the run, in the order it happens: set position,
block clearance, first step, the 10 m splits, top speed, finish. The averages that need
the whole run (stride length and frequency) come last. The returned timeline is sorted by
when each event actually happened for THIS runner (top speed can come before 50 m).

Conventions (shared with sprint_report.py so the numbers agree):
  * time 0 = frame 0 of the cleaned trial: the set position, the last frame the hands
    are on the track. Every later time is "seconds since first movement".
  * distance = forward travel of the pelvis since frame 0 (`d_pelvis` in sprint_report).
  * speed = 3-D thorax speed, smoothed with the penalised B-spline; top speed is the peak
    of that smooth curve (the raw peak is mostly marker noise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import sprint_pipeline as SP
import stride as ST

SPLITS_M = (10, 20, 30, 40, 50)
FINISH_M = 60


@dataclass
class RunSummary:
    runner: str
    trial: dict              # the cleaned trial (SP.clean_for_angles)
    contacts: list           # stride.Contact, both feet, in time order
    blocks: ST.BlockStart
    speed: np.ndarray        # smoothed speed per frame, m/s
    distance: np.ndarray     # pelvis forward travel per frame, m
    timeline: pd.DataFrame   # event, frame, time_s, distance_m, value
    headline: dict


def summarize(runner):
    t = SP.clean_for_angles(runner)
    raw, mk, fs = t["raw64"], t["mk"], t["fs"]
    rows = []

    def event(name, frame, value=None):
        rows.append({"event": name, "frame": frame, "time_s": frame / fs,
                     "distance_m": distance[frame], "value": value})

    distance = mk[:, SP.IDX_PELVIS, SP._AP] - mk[0, SP.IDX_PELVIS, SP._AP]
    speed = SP.peak_velocity_smoothed(SP.compute_velocity(mk, fs, raw64=raw)[0])[1]

    # ── 1. Set position: the run starts here (frame 0) ────────────────────────
    event("set position (first movement)", 0)

    # ── 2. Block clearance: each foot's last frame on its pedal ───────────────
    contacts = ST.all_contacts(raw, fs)
    blocks = ST.block_start(raw, contacts)
    event(f"rear foot ({blocks.rear_foot}) leaves block", blocks.rear_clearance)
    event(f"front foot ({blocks.front_foot}) leaves block", blocks.front_clearance)

    # ── 3. First step on the track ───────────────────────────────────────────
    ground = [c for c in contacts if not c.on_blocks]
    step1 = ground[0]
    event(f"step 1 touchdown ({step1.foot})", step1.touchdown,
          f"block flight {(step1.touchdown - blocks.front_clearance) / fs:.3f} s")

    # ── 4. Splits: first frame the pelvis has travelled 10, 20, ... m ─────────
    for m in SPLITS_M:
        event(f"{m} m", int(np.argmax(distance >= m)))

    # ── 5. Top speed: peak of the smooth speed curve ─────────────────────────
    peak = int(np.argmax(speed))
    steps_to_peak = sum(c.touchdown <= peak for c in ground)
    event("top speed", peak, f"{speed[peak]:.2f} m/s after {steps_to_peak} steps")

    # ── 6. Finish: pelvis has travelled 60 m ─────────────────────────────────
    finish = int(np.argmax(distance >= FINISH_M)) if distance[-1] >= FINISH_M else len(distance) - 1
    event(f"finish ({FINISH_M} m)", finish)

    # ── 7. Whole-run averages: strides from step 1 to the finish ─────────────
    #    A stride is touchdown to the next touchdown of the SAME foot (two steps).
    tds = {f: [c.touchdown for c in ground if c.foot == f and c.touchdown <= finish] for f in "RL"}
    strides = [(a, b) for f in "RL" for a, b in zip(tds[f], tds[f][1:])]
    length = np.mean([distance[b] - distance[a] for a, b in strides])
    freq = np.mean([fs / (b - a) for a, b in strides])

    # The blocks above follow the usual order of a run, but runners differ (top speed can
    # come before 50 m), so the table is sorted by when each event actually happened.
    timeline = pd.DataFrame(rows).sort_values("frame", kind="stable").round({"time_s": 3, "distance_m": 2})
    headline = {
        "runner": t["pid"],
        "time_to_60m_s": round(finish / fs, 3),
        "top_speed_ms": round(float(speed[peak]), 2),
        "time_to_top_speed_s": round(peak / fs, 3),
        "distance_at_top_speed_m": round(float(distance[peak]), 1),
        "steps_to_top_speed": steps_to_peak,
        "steps_to_60m": sum(c.touchdown <= finish for c in ground),
        "avg_stride_length_m": round(float(length), 2),
        "avg_stride_frequency_hz": round(float(freq), 2),   # same foot to same foot
        "avg_step_frequency_hz": round(float(2 * freq), 2),  # = coaching "stride rate"
        "block_time_s": round(blocks.front_clearance / fs, 3),
        # QA: in running the feet must alternate. Each repeat is a contact the detector missed.
        "missed_contacts": sum(a.foot == b.foot for a, b in zip(ground, ground[1:])),
    }
    return RunSummary(t["pid"], t, contacts, blocks, speed, distance, timeline, headline)


if __name__ == "__main__":
    run = summarize("SB80")
    print(run.timeline.to_string(index=False))
    print(run.headline)
    assert list(run.timeline.frame) == sorted(run.timeline.frame), "events must be chronological"
    assert run.headline["missed_contacts"] == 0
