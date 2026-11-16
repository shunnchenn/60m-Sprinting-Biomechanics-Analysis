"""
sprint_outputs.py — sections 7 and 8: every figure, every MP4.

The analysis lives in `sprint_pipeline.py` and is walked step by step in
`sprint_analysis.ipynb`. That notebook is where the testing happens — each stage
comes back as a DataFrame you can take the head and the shape of. So this half
has no reason to be interactive. It has one job:

    python sprint_outputs.py          # every figure, every athlete's MP4

`run_all_outputs()` does the same thing from a notebook or a REPL, and every
renderer below can still be called on its own.

    Figures/                 the cohort-wide PNGs
    Figures/Block Start/     one block-exit kinogram per athlete, all 30
    MP4s/                    the three group animations
    MP4s/Trial Runs/         one three-panel video per athlete, all 30

Anything produced once PER ATHLETE gets its own Title Case folder, so a set of
thirty stays together instead of burying the handful of figures that describe
the cohort as a whole. `cohort_dir()` is the one place that names them.

The three group animations are all top 3 against bottom 3, and they answer three
different questions: one stride at top speed, the blocks through 10 steps
(~16 m), and the whole run with the entire 62.5 m held in one fixed view.

Needs ffmpeg on PATH.

--------------------------------------------------------------------------
Why the animation code is written the way it is
--------------------------------------------------------------------------
The earlier renderers looked choppy for four separate reasons, and all four are
fixed here rather than papered over with a higher bitrate.

1. FRAME RATE.      Trials were rendered at 20 fps from every 2nd or 3rd sample
                    of a 60 Hz capture, so at best 20 distinct body positions
                    reached the eye each second. Now every captured sample is
                    drawn and the output runs at the capture rate — 60 fps, real
                    time. `every=N` still exists for a quick preview and drops
                    the frame rate to match, so timing stays honest.
2. LAYOUT JITTER.   The group animations used constrained layout with a title
                    that changed every frame. Constrained layout re-solves on
                    every draw, so the axes crept a pixel or two whenever the
                    title text changed width — the whole picture shivered. The
                    axes are now placed at fixed rectangles and the changing
                    text sits at a fixed position, so nothing can move.
3. CAMERA JITTER.   The side-on camera tracked the T12 marker, which oscillates
                    with every stride, so the background slid back and forth
                    under the athlete. The camera now follows a smoothed track.
4. LOOP SEAM.       A stride cycle is sampled 0-100 % inclusive, and 100 % is
                    the same instant as 0 % of the next cycle. Rendering all 101
                    samples therefore showed one frame twice at every loop join.
                    The last sample is now dropped, so the loop is seamless.

Rendering every frame instead of every third is 3x the frames, so the drawing
had to get cheaper to match. Two changes pay for it: the artists are built once
and updated per frame (`_Skeleton`, `set_data`) instead of clearing and
re-plotting the axes, and frames go to ffmpeg as raw RGBA straight off the Agg
canvas instead of being PNG-encoded and decoded again.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# NOTE: the backend is deliberately left alone. Forcing "Agg" here renders the
# MP4 frames fine but silently stops figures displaying inline in a notebook.
# The video renderers below sidestep the question entirely: they build a bare
# Figure with an Agg canvas attached by hand and never touch pyplot.
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

import sprint_pipeline as SP
from project_paths import FIGURES_DIR, MP4_DIR
from sprint_pipeline import (
    C3D_DIR, EXCLUDED_PIDS, _AP, _VERT, MK_PELVIS_ALL, MK_T12, MK_R_HEEL, MK_L_HEEL, MK_R_TOE, MK_L_TOE,
    R_HEEL, L_HEEL, COL_FAST, COL_SLOW, PC_MINUS, PC_PLUS, INK, COL_DOT, LEFT_ALPHA, ANGLE_NAMES,
    clean_for_angles, height_scale_for, find_top_speed_strides, find_first_steps, smooth_angle_curves,
    run_angle_fpca, fast_vs_slow_angles,
)

def _sprint_report():
    """Import the legacy report module, which now lives in (Deprecated)/."""
    try:
        import sprint_report
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent / "(Deprecated)"))
        import sprint_report
    return sprint_report


# ── Output directories ──────────────────────────────────────────────────────
# The MP4s get their own folder. The output run writes one per athlete, so
# thirty-odd videos would otherwise be mixed in with the PNGs.
FIG_OUT = FIGURES_DIR; FIG_OUT.mkdir(parents=True, exist_ok=True)
MP4_OUT = MP4_DIR;     MP4_OUT.mkdir(parents=True, exist_ok=True)

# Anything produced once PER ATHLETE goes in its own Title Case subfolder, so a
# cohort-wide set stays together and never buries the handful of figures that
# describe the cohort as a whole.
def cohort_dir(parent, name):
    """`Figures/Block Start/`-style folder for one all-30-athletes output set.

    Title Case, first letter of every word capitalised, created on demand.
    """
    out = Path(parent) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


DRAW_NOTE = ("Skeletons drawn at 1.75 m stature for display; positions, distances "
             "and speeds are unscaled")


def draw_at_175(pts, pid, floor):
    """Redraw a body at 1.75 m about its own pelvis (horizontal) and the floor (vertical).

    DISPLAY ONLY. The pelvis stays where it really was, so step length, distance
    and speed are untouched; only the body around it is resized, so athletes of
    different heights look alike on screen. `pts` is (..., 64, 3), axes AP/ML/VERT.
    """
    hs  = height_scale_for(pid.split("-")[0].strip(), 1.75)
    out = np.array(pts, float, copy=True)
    pel = out[..., MK_PELVIS_ALL, :].mean(axis=-2, keepdims=True)
    out[..., :2] = pel[..., :2] + (out[..., :2] - pel[..., :2]) * hs
    out[..., 2] = floor + (out[..., 2] - floor) * hs
    return out


def cohort_pids():
    """Every participant ID with a trial on disk, excluded ones dropped."""
    seen = []
    for fpath in sorted(C3D_DIR.glob("*.c3d")):
        pid = fpath.stem.split("-")[0].strip()
        if pid not in EXCLUDED_PIDS and pid not in seen:
            seen.append(pid)
    return seen


# ==========================================================================
# 7 · Figures
# ==========================================================================

# The skeleton is drawn from joint centres, not from the 64-marker mesh: the mesh's
# triangles cross each other in a side view and the body becomes unreadable.
# Each joint is the mean of the markers that sit on it (0-based Xsens indices).
JOINTS = [MK_PELVIS_ALL, [MK_T12], [15], [16, 17, 18, 19],          # 0 pelvis, 1 T12, 2 neck, 3 head
          [20], [21], [22, 23], [24, 25], [26, 27], [29, 30],      # 4/5 shoulders, 6/7 elbows, 8/9 wrists
          [32, 34], [35, 37], [38], [42], [39, 40], [43, 44],      # 10/11 hands, 12/13 hips, 14/15 knees
          [47, 48], [50, 51], [52], [58], [57], [63]]              # 16/17 ankles, 18/19 heels, 20/21 toes
_W = np.zeros((len(JOINTS), 64))
for j, mks in enumerate(JOINTS):
    _W[j, mks] = 1 / len(mks)
HEAD = 3

# Chris V. 64-marker bone polylines (0-based, verbatim from MATLAB
# `xsens_scr_video.m` via the archived sprint_animation.py). These index the
# RAW 64 markers, not the joint centres above, and give the dense mesh look of
# the original ensemble figures.
AXIAL_IDX = [
    [1,2,4,3,1,5,6,2,6,7,5,7,8,9,10,13,15,20,11,14,10,14,20,14,21,14,11,21,15],
    [21,10], [20,10], [16,17,18,16,19,17,19,18,19,15]]
RIGHT_IDX = [
    [20,22,23,28,23,20], [20,28,26,28,27,26], [23,26,23,27,26], [22,26,22,27,26],
    [26,32,27,32,33,32,34,32,33,26],
    [38,39,38,40,38,41,40,39,41,46,39,46,40,46],
    [46,47,46,48], [39,47,39,48], [40,47,40,48],
    [47,48,52,47,52,53,52,54,52,55,56,55,57,52]]
LEFT_IDX = [
    [21,24,25,24,31,21], [21,25],
    [29,30,24,29,24,30,24,25,29,25,30,25],
    [31,29,30,29,31,30,31],
    [29,35,29,36,29,37,35,36,30,35,30,36,30,37,30],
    [42,43,42,44,42,45,44,43,45],
    [49,45,49,44,49,43,49], [49,50,49,51,49],
    [43,50,43,51,43], [44,50,44,51,44],
    [50,51,50,58,51,58,59,58,59,58,61,58,60,61,62,61,63]]


def joints(pts):
    """(..., 64, 3) marker positions -> (..., 22, 3) joint centres."""
    return np.einsum("jm,...md->...jd", _W, pts)


def _draw_raw64(ax, frame, color, alpha=0.95, lw=1.6, label=None, floor=None,
                left_alpha=0.35):
    """Draw one stick figure (joint centres of a 64-marker frame) in the sagittal plane, in metres.

    Same bones as the MP4 exporter, so the anatomy in the figures and in the
    videos match. The LEFT limbs are drawn faint: in a sagittal view the two
    sides sit on top of each other and cross constantly, so without a contrast
    it is impossible to tell which leg is which.
    """
    pts = frame.copy()
    centre = pts[MK_PELVIS_ALL].mean(axis=0).copy()
    centre[_VERT] = 0.0
    pts = pts - centre                                    # centre horizontally
    if floor is not None:
        pts[:, _VERT] -= floor                            # one fixed floor

    # Trunk and right side solid, left side faded. The polylines index the raw
    # 64 markers (Chris V. mesh), so no joint-centre reduction here.
    first = True
    for polys, a_mul in ((AXIAL_IDX, 1.0), (RIGHT_IDX, 1.0), (LEFT_IDX, left_alpha)):
        for poly in polys:
            ax.plot(pts[poly, _AP], pts[poly, _VERT], color=color, lw=lw,
                    alpha=alpha * a_mul, solid_capstyle="round", solid_joinstyle="round",
                    label=label if first else None)
            first = False


def pc_raw64_extremes(cycles_by_pid, fpca, pc=1, k=2.0):
    """Marker-space skeletons for mean +- k SD of a component.

    The component lives in ANGLE space. To draw it as a real skeleton, each
    marker coordinate at each phase is regressed on the participants' PC scores,
    and the fitted line is evaluated at +-k standard deviations. So this is a
    faithful projection of the angle component into marker space, not a second
    PCA — the analysis is still the one method, done once.

    Returns (minus, plus), each (n_points, 64, 3).
    """
    pids   = fpca["pids"]
    X      = np.stack([cycles_by_pid[p] for p in pids])       # (n_pid, pts, 64, 3)
    scores = fpca["scores"][:, pc - 1]

    # 1. Least-squares slope of every coordinate against the PC score.
    s_c   = scores - scores.mean()
    mean  = X.mean(axis=0)
    slope = np.tensordot(s_c, X - mean, axes=(0, 0)) / np.sum(s_c ** 2)

    # 2. Step out k standard deviations either side of the mean posture.
    step = k * float(scores.std())
    return mean - step * slope, mean + step * slope


def plot_pc_anatomy(cycles_by_pid, fpca, peak_vel, pc=1, k=2.0,
                    phases=(0, 25, 50, 75), title_extra="", fname=None,
                    save=True):
    """Draw a component as two stick figures across the cycle: the posture of a
    runner scoring 2 SD below the mean on it, and one scoring 2 SD above.

    Same skeleton construction as the MP4s.
    """
    minus, plus = pc_raw64_extremes(cycles_by_pid, fpca, pc, k)

    # 1. The two ends of the component get neutral colours, not faster/slower blue
    #    and red: a component's link to speed is a separate (and usually weak) fact,
    #    so it goes in the title as r rather than being painted onto the skeletons.
    r = float(np.corrcoef(fpca["scores"][:, pc - 1],
                          np.array([peak_vel[p] for p in fpca["pids"]]))[0, 1])
    minus_col, plus_col = PC_MINUS, PC_PLUS

    # 2. One floor for every panel, taken across both skeletons, so the figures
    #    sit on the same ground instead of each floating to its own.
    foot_ids = [MK_R_HEEL, MK_L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = min(float(minus[:, foot_ids, _VERT].min()),
                float(plus[:,  foot_ids, _VERT].min()))

    fig, axes = plt.subplots(1, len(phases), figsize=(4.1 * len(phases), 6.8),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ph in zip(axes, phases):
        _draw_raw64(ax, minus[ph], minus_col, floor=floor,
                    label=f"Low PC{pc} score (mean − {k:g} SD)")
        _draw_raw64(ax, plus[ph],  plus_col, floor=floor,
                    label=f"High PC{pc} score (mean + {k:g} SD)")
        ax.axhline(0.0, color="#8a6d3b", lw=1.4, alpha=0.6, zorder=0)
        ax.text(0.03, 0.97, f"{ph}% of the stride", transform=ax.transAxes, va="top", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("Fore–aft position (m)")
    axes[0].set_ylabel("Vertical position (m)")

    imp  = np.linalg.norm(fpca["loadings"][pc - 1], axis=0)
    side = lambda n: {"_R": "right ", "_L": "left "}.get(n[-2:], "") + n.removesuffix("_R").removesuffix("_L").replace("_", " ")
    top3 = [side(fpca["names"][i]) for i in np.argsort(-imp)[:3]]
    fig.suptitle(
        f"Functional PC{pc} at ±{k:g} SD{title_extra}: "
        f"{fpca['explained_var'][pc - 1]:.1f}% of variance, "
        f"r = {r:+.2f} with peak velocity",
        fontsize=12)
    # The "which joints" note rides as the legend TITLE. A separate fig.text in
    # the bottom margin lands on top of the legend, because constrained layout
    # puts both in the same strip; as the title it gets its own line.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False, fontsize=11,
               title=f"Largest loadings: {', '.join(top3)}; faint limbs = left side\n{DRAW_NOTE}",
               title_fontproperties={"style": "italic", "size": 10.5})

    if save:
        out = FIG_OUT / (fname or f"pc{pc}_anatomy.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out      # left open on purpose: a notebook then shows it once
    return fig


def plot_pc_loading_ranking(fpca, pc=1, save=True):
    """Which joints does this component actually use? A ranked bar chart.

    Importance for one angle is the size (L2 norm) of that angle's loading
    across the whole cycle. A long bar means the component varies a lot in that
    joint; a short bar means the joint barely participates.
    """
    k          = pc - 1
    loading    = fpca["loadings"][k]                 # (points, angles)
    importance = np.linalg.norm(loading, axis=0)
    order      = np.argsort(importance)              # ascending, so biggest ends on top
    names      = [angle_label(fpca["names"][i]) for i in order]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.barh(range(len(order)), importance[order], color="0.55")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Loading magnitude")
    ax.set_title(f"Joint loadings on functional PC{pc} "
                 f"({fpca['explained_var'][k]:.1f}% of variance)")

    if save:
        out = FIG_OUT / f"pc{pc}_loading_ranking.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out      # left open on purpose: a notebook then shows it once
    return fig


ANGLE_PAIRS = [("hip", "hip_R", "hip_L"), ("knee", "knee_R", "knee_L"),
               ("ankle", "ankle_R", "ankle_L"),
               ("shoulder", "shoulder_R", "shoulder_L"),
               ("elbow", "elbow_R", "elbow_L"),
               ("trunk lean", "trunk_lean", None),
               ("thigh separation", "thigh_separation", None),
               ("arm separation", "arm_separation", None)]


def plot_fast_vs_slow(cohort, peak_vel, n=5, save=True):
    """Angle curves through the cycle, fast group against slow group.

    Eight panels, one per joint. Right side solid, left side dotted and shifted
    half a cycle so the two lie on top of each other - any remaining gap is a
    genuine left-right asymmetry rather than the legs simply alternating.
    """
    _table, fast, slow, _, _ = fast_vs_slow_angles(cohort, peak_vel, n)
    x = np.linspace(0, 100, fast.shape[0])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (title, r_name, l_name) in zip(axes.ravel(), ANGLE_PAIRS):
        for name, style, shift in ((r_name, "-", 0), (l_name, ":", 50)):
            if name is None:
                continue
            j = ANGLE_NAMES.index(name)
            # The left leg does the SAME thing half a cycle later, so shifting it
            # by 50 % lays it on top of the right. (Flipping it vertically only
            # looks right because these curves are near-sinusoidal - it gives
            # r = 0.42 at the ankle where the shift gives 1.00.) Any gap that
            # survives the shift is a real left-right asymmetry.
            s = np.roll(slow[:, j], shift)
            f = np.roll(fast[:, j], shift)
            # Rolling wraps the array, which leaves a seam where the end meets
            # the start. Draw the two halves as separate lines so no stroke is
            # drawn across that join.
            cuts = [slice(None)] if shift == 0 else [slice(0, shift), slice(shift, None)]
            for k, sl in enumerate(cuts):
                lbl = "" if k else ("" if shift == 0 else ", left side")
                ax.plot(x[sl], s[sl], color=COL_SLOW, lw=2, ls=style,
                        label=(f"{n} slowest" + lbl) if k == 0 else None)
                ax.plot(x[sl], f[sl], color=COL_FAST, lw=2, ls=style,
                        label=(f"{n} fastest" + lbl) if k == 0 else None)
            if shift == 0:
                ax.fill_between(x, s, f, color="#999999", alpha=0.25)
        ax.set_title(_cap(title), fontsize=11)
        ax.set_xlabel("Stride cycle (%)", fontsize=9)
        ax.set_ylabel("Angle (°)", fontsize=9)
        ax.tick_params(labelsize=8); ax.grid(alpha=0.25)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False, fontsize=11)
    fig.suptitle(
        f"Joint angles over the stride cycle, {n} fastest vs {n} slowest runners\n"
        f"Solid = right side; dotted = left side shifted by 50% of the cycle",
        fontweight="bold", fontsize=12)

    if save:
        out = FIG_OUT / "fast_vs_slow_angles.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out      # left open on purpose: a notebook then shows it once
    return fig


def plot_event_strip(raw64, events, fs, ncols=None):
    """One skeleton per event, left to right, on a shared floor: [(label, frame), ...].

    Used by the utility notebooks to check event detection by eye. A frame of None
    (event not found) leaves its panel blank.
    """
    ncols = ncols or len(events)
    nrows = -(-len(events) // ncols)
    found = [f for _l, f in events if f is not None]
    foot_ids = [MK_R_HEEL, MK_L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = float(raw64[found][:, foot_ids, _VERT].min())
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 3.6 * nrows),
                             sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    for ax, (label, f) in zip(axes.ravel(), events):
        ax.axhline(0.0, color="#8a6d3b", lw=1.2, alpha=0.6)
        ax.set_title(label if f is None else f"{label}\nframe {f} ({f / fs:.2f} s)", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
        if f is not None:
            _draw_raw64(ax, raw64[f], INK, floor=floor, lw=1.4)
    for ax in axes.ravel()[len(events):]:
        ax.axis("off")
    return fig


def plot_block_exit_steps(trial, save=True):
    """Block exit and the first three steps, as five sagittal skeletons.

    Reads left to right: the athlete set in the blocks, the moment the hands
    leave, then each of the first three foot contacts. This is the phase the
    old velocity-only start was cutting off.

    Left limbs are faint, same convention as every other skeleton figure here.
    """
    t   = clean_for_angles(trial)
    raw = t["raw64"]

    # 1. Frame 0 IS the set position now, because clean_for_angles starts there.
    #    Block exit is the FRONT foot leaving - the last contact with the blocks.
    #    The hands lift first (about 4 frames in) but that is barely a change of
    #    posture; the front toe leaving is the moment the athlete is airborne
    #    from the blocks, and it lands between the hands lifting and step 1.
    #    25 mm of rise is the same threshold the older kinogram code settled on.
    front_toe = (MK_R_TOE if raw[0, MK_R_HEEL, _AP] > raw[0, MK_L_HEEL, _AP]
                 else MK_L_TOE)
    toe_z = raw[:, front_toe, _VERT]
    off   = np.where(toe_z > toe_z[0] + 0.025)[0]
    exit_f = int(off[0]) if len(off) else 1

    # 2. The first three steps, from the detector used by the analysis.
    steps, _ = find_first_steps(t["mk"], n_steps=3)
    cols = [("Set — all four down", 0), ("Block exit (front foot)", exit_f)]
    cols += [(f"Step {i+1} contact", a) for i, (a, _b) in enumerate(steps)]

    # 3. One floor for every panel so the skeletons share a ground line.
    foot_ids = [MK_R_HEEL, MK_L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = float(raw[[f for _n, f in cols]][:, foot_ids, _VERT].min())

    fig, axes = plt.subplots(1, len(cols), figsize=(3.5 * len(cols), 5.6),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, f) in zip(axes, cols):
        _draw_raw64(ax, draw_at_175(raw[f], t["pid"], floor), INK, floor=floor, lw=1.7)
        ax.axhline(0.0, color="#8a6d3b", lw=1.4, alpha=0.6, zorder=0)
        pel_x = raw[f, MK_PELVIS_ALL, _AP].mean()
        ax.set_title(f"{name}\n{f / t['fs']:.2f} s, pelvis at {pel_x:.2f} m", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("Fore–aft position (m)")
    axes[0].set_ylabel("Vertical position (m)")
    fig.suptitle(f"{t['pid']}: block exit and first three steps"
                 f" (peak velocity {t['peak_vel_ms']:.2f} m/s)\n"
                 f"Solid = right side; faint = left side\n{DRAW_NOTE}",
                 fontweight="bold", fontsize=12)

    if save:
        # One per athlete, so it goes in the cohort folder rather than loose
        # among the figures that describe the whole cohort.
        out = cohort_dir(FIG_OUT, "Block Start") / f"block_exit_{t['pid']}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out      # left open on purpose: a notebook then shows it once
    return fig


# --------------------------------------------------------------------------
# Figures for the ML stage
# --------------------------------------------------------------------------

def plot_velocity_fit(c, pid=None, save=True):
    """The target being cleaned: raw numerical derivative against its B-spline (c = sprint_report cohort cache)."""
    pid = pid or sorted(c["vel"])[0]
    (v_sm, fs), v_raw = c["vel"][pid], c["vel_raw"][pid]
    t = np.arange(len(v_raw)) / fs

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.7, 1]})

    ax = axes[0]
    ax.plot(t, v_raw, color="#999999", lw=0.9, label="Raw (numerical derivative)")
    ax.plot(t, v_sm, color=COL_FAST, lw=2.2, label="Penalised B-spline fit")
    ax.axhline(v_raw.max(), color="#999999", ls=":", lw=1)
    ax.axhline(v_sm.max(),  color=COL_FAST, ls=":", lw=1)
    ax.annotate(f"{v_raw.max() - v_sm.max():+.2f} m/s",
                xy=(t[-1] * 0.72, (v_raw.max() + v_sm.max()) / 2),
                fontsize=10, color=INK)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Pelvis speed (m/s)")
    ax.set_title(f"{pid}: raw and smoothed pelvis speed", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    infl = c["qa"].v_noise_ms.values
    ax.hist(infl, bins=12, color="0.6", edgecolor="white")
    ax.axvline(infl.mean(), color="k", ls="--", lw=1.5,
               label=f"Mean {infl.mean():+.2f} m/s")
    ax.set_xlabel("Raw minus smoothed peak velocity (m/s)"); ax.set_ylabel("Runners")
    ax.set_title("Raw vs smoothed peak velocity, all runners", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    if save:
        out = FIG_OUT / "velocity_bspline_fit.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out
    return fig


# ==========================================================================
# 8 · Animation
# ==========================================================================

# ── The frame factory ───────────────────────────────────────────────────────

def _agg_figure(figsize, dpi):
    """A figure with an Agg canvas attached by hand — pyplot never sees it.

    Two reasons. It cannot disturb whatever backend a notebook is using, and it
    gives direct access to the RGBA buffer, which is what makes `_MP4Writer`
    cheap. Nothing has to be closed afterwards either: with no pyplot registry
    holding a reference, the figure is collected like any other object.
    """
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    return fig


class _MP4Writer:
    """Raw RGBA frames straight from the Agg canvas into ffmpeg.

    The old writer saved every frame as a PNG in memory and handed ffmpeg the
    compressed bytes, which then had to decode them again. Compressing a frame
    only to decompress it one pipe later costs about 60 ms a frame and buys
    nothing, and at 60 fps that alone was most of the render time. Raw frames
    skip both halves.
    """

    def __init__(self, fig, out_path, fps):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not on PATH — needed to write MP4s")
        self.fig = fig
        self.path = Path(out_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Draw once to learn the exact buffer size. Taking it from the buffer
        # rather than from figsize*dpi keeps this correct whatever rounding
        # matplotlib does.
        fig.canvas.draw()
        h, w, _ = np.asarray(fig.canvas.buffer_rgba()).shape

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
            "-framerate", str(fps), "-i", "-",
            "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-preset", "veryfast",
            "-vf", "scale=ceil(iw/2)*2:ceil(ih/2)*2",   # yuv420p needs even dims
            "-r", str(fps),
            "-g", str(max(int(fps) * 2, 2)),
            "-movflags", "+faststart",                  # starts playing sooner
            str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.n_frames = 0

    def add(self):
        """Render the current state of the figure as one frame."""
        self.fig.canvas.draw()
        try:
            self.proc.stdin.write(self.fig.canvas.buffer_rgba())
        except BrokenPipeError:
            pass          # ffmpeg died; close() reports the return code
        self.n_frames += 1

    def close(self):
        """Always reached, even if a frame fails - the original leaked the process."""
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc} writing {self.path.name}")
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _Skeleton:
    """One skeleton's worth of Line2D artists, built once and moved per frame.

    Every renderer used to clear its axes and re-plot ~25 polylines each frame.
    Clearing throws away the artists, the axis limits, the grid and the labels,
    and all of it is rebuilt from scratch — which is both slow and the reason
    anything else on the axes had to be redrawn too. Here the lines are created
    once and only their data changes.
    """

    def __init__(self, ax, axial_c, right_c, left_c, lw=1.2,
                 left_alpha=1.0, dots=False, zorder=2):
        self.lines = []
        for polys, colour, alpha in ((AXIAL_IDX, axial_c, 1.0),
                                     (RIGHT_IDX, right_c, 1.0),
                                     (LEFT_IDX,  left_c,  left_alpha)):
            for poly in polys:
                line, = ax.plot([], [], color=colour, lw=lw, alpha=alpha,
                                solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
                self.lines.append((line, poly))
        self.dots = None
        if dots:
            self.dots = ax.scatter([], [], c=COL_DOT, s=10, linewidths=0,
                                   zorder=zorder + 1)

    def set_frame(self, pts, h=_AP, v=_VERT):
        """Move the skeleton to one frame of 64 marker positions (mesh bones)."""
        for line, poly in self.lines:
            line.set_data(pts[poly, h], pts[poly, v])
        if self.dots is not None:
            self.dots.set_offsets(np.column_stack([pts[:, h], pts[:, v]]))


def _smooth_track(x, fs, win_s=0.5):
    """Centred moving average, for camera tracks only.

    The camera followed a single marker, and every marker on a sprinter
    oscillates once per stride. Panning with that oscillation slides the
    background back and forth under the athlete about three times a second,
    which reads as judder even though the skeleton itself is perfectly smooth.
    Averaging over half a second leaves the travel and removes the wobble.

    This is for the VIEW only. No measured quantity is smoothed here.
    """
    w = max(int(round(win_s * fs)) | 1, 3)          # odd, so it stays centred
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(w) / w, mode="valid")


# ── One trial, three panels ─────────────────────────────────────────────────

def render_video(c3d_path, out_path, fps=None, dpi=100, every=1) -> str:
    """Three-panel animation of one trial, written to `out_path` as MP4.

    `every`=N renders every Nth captured frame. It defaults to 1 — EVERY sample
    — because skipping frames was the main reason these videos looked choppy: a
    60 Hz capture rendered every 3rd frame shows 20 body positions a second.

    `fps` defaults to `fs / every`, so playback is real time whatever `every`
    is set to. Leave it alone unless you want deliberate slow motion.
    """
    # Same cleaning and speed as the analysis (workbook v_peak): clean_for_angles + smoothed pelvis speed.
    tr       = clean_for_angles(Path(c3d_path))
    pid, pts, fs = tr["pid"], tr["raw64"], tr["fs"]
    vel      = SP.peak_velocity_smoothed(SP.compute_velocity(tr["mk"], fs, raw64=pts)[0])[1]   # m/s
    n_frames = len(pts)
    time     = np.arange(n_frames) / fs
    fps      = float(fs) / every if fps is None else fps

    max_vel_mps = float(vel.max())
    peak_frame  = int(np.argmax(vel))
    peak_time   = time[peak_frame]

    t12_x       = pts[:, MK_T12, 0]
    pel_x       = pts[:, MK_PELVIS_ALL, 0].mean(1) - pts[0, MK_PELVIS_ALL, 0].mean()
    fin_idx     = np.where(pel_x >= 60.0)[0]                 # the run is trimmed at 62 m, so time to 60 m of pelvis travel
    finish_time = time[fin_idx[0]] if fin_idx.size else time[-1]

    # The side-on camera follows a SMOOTHED T12 track. T12 sways forward and
    # back within every stride, so panning with the raw marker drags the grid
    # and the ground back and forth under a runner who is in fact moving evenly.
    cam_x = _smooth_track(t12_x, fs, win_s=0.5)

    # Trail colours: pelvis speed on one blue ramp, light (slow) to dark (fast), one colour per segment.
    traj     = pts[:, MK_T12, :2]
    segs     = np.stack([traj[:-1], traj[1:]], axis=1)       # (n-1, 2, 2)
    seg_rgba = cm.Blues(0.25 + 0.75 * vel[:-1] / max(max_vel_mps, 1e-9))     # (n-1, 4)
    y_lim_path = (float(pts[:, :, 1].min()), float(pts[:, :, 1].max()))
    lane_y     = max(abs(y_lim_path[0]), abs(y_lim_path[1])) * 1.2

    draw = draw_at_175(pts, pid, float(pts[:, :, 2].min()))   # side-on body only; track and speed stay true

    fig = _agg_figure((12.78, 6.42), dpi)
    fig.text(0.5, 0.008, DRAW_NOTE, ha="center", va="bottom", fontsize=8, style="italic")
    ax_vel  = fig.add_axes([0.06, 0.56, 0.42, 0.35])
    ax_side = fig.add_axes([0.55, 0.56, 0.42, 0.35])
    ax_path = fig.add_axes([0.05, 0.10, 0.92, 0.32])
    fig.suptitle(f"{pid}  •  Max V = {max_vel_mps:.2f} m/s  •  "
                 f"Time to 60 m (pelvis) = {finish_time:.2f} s",
                 fontsize=12, fontweight="bold", y=0.97)

    # ── Everything that never changes is drawn ONCE ──────────────────────────
    # Panel 1: velocity curve
    ax_vel.plot(time, vel, color="k", lw=1.4)
    ax_vel.scatter([peak_time], [vel[peak_frame]],
                   c="white", s=80, edgecolors=INK, zorder=5)
    now_dot, = ax_vel.plot([], [], "o", c=INK, ms=9,
                           markeredgecolor="k", zorder=6)
    info = ax_vel.text(0.98, 0.04, "", transform=ax_vel.transAxes,
                       fontsize=9, ha="right", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor="#888888", lw=0.6, alpha=0.9))
    ax_vel.set_xlabel("Time (s)", fontsize=9)
    ax_vel.set_ylabel("Pelvis speed (m/s)", fontsize=9)
    ax_vel.set_xlim(0, time.max())
    ax_vel.set_ylim(0, vel.max() * 1.08)
    ax_vel.tick_params(labelsize=8)
    ax_vel.grid(True, alpha=0.3)

    # Panel 2: sagittal skeleton, camera-tracked
    skel = _Skeleton(ax_side, INK, INK, INK, lw=1.4, left_alpha=LEFT_ALPHA, dots=True)
    ax_side.set_ylim(draw[:, :, 2].min() - 0.05, draw[:, :, 2].max() + 0.05)
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.set_xlabel("Fore–aft position (m)", fontsize=9)
    ax_side.set_ylabel("Vertical position (m)", fontsize=9)
    ax_side.tick_params(labelsize=8)
    ax_side.grid(True, alpha=0.3)

    # Panel 3: top-down track
    ax_path.axhspan(-lane_y, lane_y, color="#f5f5f5", zorder=0)
    ax_path.axvline(0, color="#888888", lw=1, ls="--")
    ax_path.axvline(60.0, color="k", lw=2)
    ax_path.text(60.0, y_lim_path[1] * 0.85, " 60 m",
                 fontsize=9, fontweight="bold")
    # The whole trail is one LineCollection, built once. Only the alpha column
    # changes per frame, which is what makes it grow behind the athlete without
    # re-drawing hundreds of little line segments every time.
    trail = LineCollection(segs, linewidths=2.5, capstyle="round", zorder=2)
    trail_rgba = seg_rgba.copy()
    trail_rgba[:, 3] = 0.0
    trail.set_color(trail_rgba)
    ax_path.add_collection(trail)
    peak_dot, = ax_path.plot([], [], "o", c="white", ms=11,
                             markeredgecolor="k", zorder=10)
    path_dot, = ax_path.plot([], [], "o", c=INK, ms=9,
                             markeredgecolor="k", zorder=11)
    ax_path.set_xlim(-0.5, 66)
    ax_path.set_ylim(-lane_y, lane_y)
    ax_path.set_xlabel("Forward position (m)", fontsize=9)
    ax_path.set_ylabel("Lateral position (m)", fontsize=9)
    ax_path.tick_params(labelsize=8)
    ax_path.grid(True, alpha=0.3)
    ax_path.set_title("Sprint track, coloured by pelvis speed",
                      fontsize=9, pad=2)

    writer = _MP4Writer(fig, out_path, fps)
    try:
        for n in range(0, n_frames, every):
            now_dot.set_data([time[n]], [vel[n]])
            info.set_text(f"Max V: {max_vel_mps:.2f} m/s\n"
                          f"Finish: {finish_time:.2f} s\n"
                          f"Now: {time[n]:.2f} s  ({vel[n]:.2f} m/s)")

            skel.set_frame(draw[n], h=0, v=2)
            ax_side.set_xlim(cam_x[n] - 1.5, cam_x[n] + 1.5)

            trail_rgba[:, 3] = seg_rgba[:, 3] * (np.arange(len(segs)) < n)
            trail.set_color(trail_rgba)
            path_dot.set_data([pts[n, MK_T12, 0]], [pts[n, MK_T12, 1]])
            if n >= peak_frame:
                peak_dot.set_data([pts[peak_frame, MK_T12, 0]],
                                  [pts[peak_frame, MK_T12, 1]])

            writer.add()
    finally:
        writer.close()
    return str(out_path)


def render_trial(pid, fps=None, every=1, dpi=100):
    """Convenience wrapper: participant ID in, MP4 path out."""
    matches = sorted(C3D_DIR.glob(f"{pid}*.c3d"))
    if not matches:
        raise SystemExit(f"No C3D file found for '{pid}'")
    out = cohort_dir(MP4_OUT, "Trial Runs") / f"{pid}_run.mp4"
    return Path(render_video(str(matches[0]), str(out), fps=fps, every=every,
                             dpi=dpi))


# ── Fast group against slow group, top speed ────────────────────────────────

def _analysis_cycle(pid, n_points=101):
    """One mean top-speed cycle of 64-marker positions for a participant.

    Uses the ANALYSIS cleaning, not the pipeline above, because this video is
    the animated form of pc1_anatomy.png and has to match it. Scaled to 1.75 m
    for drawing so averaging different-sized athletes does not blur the mean.
    """
    t = clean_for_angles(pid)
    windows, _ = find_top_speed_strides(t["mk"], t["peak_frame"])
    if not windows:
        return None, None

    raw  = t["raw64"] * height_scale_for(pid, 1.75)
    grid = np.linspace(0, 1, n_points)
    cycles = []
    for a, b in windows:
        seg = raw[a:b + 1]
        src = np.linspace(0, 1, len(seg))
        cycles.append(np.stack([[np.interp(grid, src, seg[:, m, ax])
                                 for ax in range(3)]
                                for m in range(seg.shape[1])]).transpose(2, 0, 1))
    return np.mean(cycles, axis=0), t["peak_vel_ms"]


def render_representative(n=3, out_path=None, fps=50, loops=4, dpi=110):
    """The `n` fastest athletes averaged into one skeleton, against the `n` slowest.

    Not one athlete each - the mean of `n`, so the comparison is of a
    representative fast sprinter against a representative slow one rather than
    two individuals who might each be unusual.

    The cycle is 101 samples of a stride that really lasts about 0.45 s, so at
    50 fps this plays at roughly a quarter speed and each frame advances the
    body by 1 % of the cycle. The old 12 fps showed the same 1 % steps four
    times more slowly, which is what made it stutter.
    """
    out_path = Path(out_path or MP4_OUT / f"representative_top{n}_vs_bottom{n}.mp4")

    # 1. Rank by real (unscaled) peak velocity.
    ranked = []
    for pid in cohort_pids():
        cyc, vel = _analysis_cycle(pid)
        if cyc is not None:
            ranked.append((vel, pid, cyc))
    ranked.sort(key=lambda r: r[0])

    slow, fast = ranked[:n], ranked[-n:]

    # 2. Average each group into one representative skeleton.
    groups = []
    for grp, colour, role in ((fast, COL_FAST, "faster"), (slow, COL_SLOW, "slower")):
        groups.append(dict(cyc=np.mean([c for _v, _p, c in grp], axis=0),
                           pids=[p for _v, p, _c in grp],
                           vel=np.mean([v for v, _p, _c in grp]),
                           colour=colour, role=role))

    # 3. One shared floor so neither skeleton floats.
    foot = [R_HEEL, L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = min(float(g["cyc"][:, foot, 2].min()) for g in groups)

    # 4. Pre-centre every frame: pelvis over the origin, feet on the floor.
    #    Doing it here rather than inside the loop keeps the per-frame work to a
    #    set_data call.
    for g in groups:
        cyc = g["cyc"].copy()
        c = cyc[:, MK_PELVIS_ALL, :].mean(axis=1)         # (points, 3)
        c[:, 2] = floor                                   # keep height, drop to floor
        g["draw"] = cyc - c[:, None, :]

    # 100, not 101: sample 100 is the same instant as sample 0 of the next
    # cycle, so rendering it would show one frame twice at every loop join.
    n_pts = groups[0]["draw"].shape[0] - 1

    # Fixed axes rectangle and fixed text positions. Constrained layout re-solves
    # on every draw, so a title that changes width - "7 %" to "100 %" - nudged
    # the axes and made the whole frame shiver.
    fig = _agg_figure((7.6, 7.4), dpi)
    ax  = fig.add_axes([0.10, 0.105, 0.86, 0.755])   # bottom clear of the x-label + note
    f, s = groups[0], groups[1]
    fig.text(0.5, 0.005, DRAW_NOTE, ha="center", va="bottom", fontsize=7.5, style="italic")
    fig.text(0.5, 0.975, "Fastest vs slowest runners, top-speed stride",
             ha="center", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.935,
             f"blue: {', '.join(f['pids'])}  ({f['vel']:.2f} m/s avg)      "
             f"red: {', '.join(s['pids'])}  ({s['vel']:.2f} m/s avg)",
             ha="center", va="top", fontsize=10)

    skels = [_Skeleton(ax, g["colour"], g["colour"], g["colour"], lw=2.0,
                       left_alpha=LEFT_ALPHA) for g in groups]
    phase = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=11,
                    fontweight="bold", ha="left", va="top")
    ax.axhline(0.0, color="#8a6d3b", lw=1.4, alpha=0.6, zorder=0)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.15, 2.05)
    ax.set_aspect("equal"); ax.grid(alpha=0.22)
    ax.set_xlabel("Fore–aft position (m)"); ax.set_ylabel("Vertical position (m)")

    writer = _MP4Writer(fig, out_path, fps)
    try:
        for _ in range(loops):
            for i in range(n_pts):
                for skel, g in zip(skels, groups):
                    skel.set_frame(g["draw"][i], h=0, v=2)
                phase.set_text(f"{i} % of the stride cycle")
                writer.add()
    finally:
        writer.close()
    return out_path


# ── Fast group against slow group, blocks through N steps ───────────────────

def _accel_sequence(pid, n_steps=10, per_step=30):
    """One athlete from the blocks through `n_steps`, on a step-anchored clock.

    Top speed could be averaged on a single stride cycle because every stride is
    a repeat of the last. Acceleration is a PROGRESSION - step 1 is not step 10 -
    so the steps are used as anchors instead: each step is resampled to
    `per_step` points, which guarantees that "step 5" is genuinely step 5 for
    every athlete however long they took to get there.

    Shape and position are separated on purpose:
      * shape    is scaled to 1.75 m so athletes of different sizes average into
                 a clean skeleton rather than a blur,
      * position is the TRUE, unscaled pelvis path, so the distance covered is
                 honest and not stretched by the rescale.

    Returns (shape, pelvis, seconds) or None.
    """
    t = clean_for_angles(pid)
    windows, _ = find_first_steps(t["mk"], n_steps=n_steps)
    if len(windows) < n_steps:
        return None

    raw   = t["raw64"]
    pelv  = raw[:, MK_PELVIS_ALL, :].mean(axis=1)          # true, unscaled
    hs    = height_scale_for(pid, 1.75)
    grid  = np.linspace(0, 1, per_step, endpoint=False)

    shapes, positions, secs = [], [], []
    for a, b in windows:
        seg_shape = (raw[a:b + 1] - pelv[a:b + 1, None, :]) * hs   # about the pelvis
        seg_pelv  = pelv[a:b + 1]
        src = np.linspace(0, 1, len(seg_shape))
        for u in grid:
            shapes.append(np.stack([[np.interp(u, src, seg_shape[:, m, ax])
                                     for ax in range(3)]
                                    for m in range(seg_shape.shape[1])]))
            positions.append(np.array([np.interp(u, src, seg_pelv[:, ax])
                                       for ax in range(3)]))
            secs.append((a + u * (b - a)) / t["fs"])
    return np.array(shapes), np.array(positions), np.array(secs)


def render_representative_accel(n=3, n_steps=10, per_step=30, out_path=None,
                                fps=50, dpi=110):
    """Blocks through `n_steps`, fast group against slow group, two panels.

    LEFT   pelvis-centred, so every difference is posture - the same view as the
           top-speed comparison.
    RIGHT  ground-referenced, so the step lengths and the gap that opens up are
           visible. The camera follows the pair.

    Both panels run on the same step-anchored clock. That is what makes the left
    panel meaningful, but it means the two groups are NOT at the same instant in
    time - the slower group takes longer to reach any given step. Elapsed time is
    printed for each group so that is never hidden.

    `per_step` is 30 rather than 20: each step is a separate resampling, so the
    number of samples per step IS the frame rate of the movement. 20 of them
    across a 0.2 s step is a coarse picture of a fast limb.
    """
    out_path = Path(out_path or MP4_OUT / f"representative_accel_top{n}_vs_bottom{n}.mp4")

    # 1. Rank by real peak velocity, using the same six subjects as top speed.
    ranked = sorted((clean_for_angles(pid)["peak_vel_ms"], pid)
                    for pid in cohort_pids())
    picks = {"fast": [p for _v, p in ranked[-n:]], "slow": [p for _v, p in ranked[:n]]}

    # 2. Average each group, shape and position separately.
    groups = []
    for key, colour, role in (("fast", COL_FAST, "faster"), ("slow", COL_SLOW, "slower")):
        seqs = [_accel_sequence(p, n_steps, per_step) for p in picks[key]]
        seqs = [s for s in seqs if s is not None]
        if not seqs:
            raise SystemExit(f"no usable {key} sequences")
        groups.append(dict(
            shape=np.mean([s[0] for s in seqs], axis=0),
            pelv =np.mean([s[1] for s in seqs], axis=0),
            secs =np.mean([s[2] for s in seqs], axis=0),
            pids=picks[key], colour=colour, role=role))

    n_pts = groups[0]["shape"].shape[0]
    # 3. One floor for both panels, from the feet across the whole sequence.
    foot = [R_HEEL, L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = min(float((g["shape"] + g["pelv"][:, None, :])[:, foot, 2].min())
                for g in groups)

    # Both panels use an equal aspect, so their widths must be in the same ratio
    # as their x-spans or the ground panel renders as a small wide strip. Fixed
    # rectangles, not constrained layout: the header text changes every frame.
    HALF_P, HALF_G = 1.3, 1.9
    fig = _agg_figure((14.5, 6.6), dpi)
    LEFT, GAP, RIGHT = 0.045, 0.055, 0.015          # margins in figure fractions
    W = (1 - LEFT - GAP - RIGHT) * np.array([HALF_P, HALF_G]) / (HALF_P + HALF_G)
    ax_p = fig.add_axes([LEFT,                  0.09, W[0], 0.74])
    ax_g = fig.add_axes([LEFT + W[0] + GAP,     0.09, W[1], 0.74])

    f, s = groups[0], groups[1]
    fig.text(0.5, 0.005, DRAW_NOTE, ha="center", va="bottom", fontsize=8, style="italic")
    fig.text(0.5, 0.985, f"Fastest vs slowest runners, blocks to step {n_steps}", ha="center", va="top",
             fontsize=13, fontweight="bold")
    header = fig.text(0.5, 0.945, "", ha="center", va="top", fontsize=10.5)

    skel_p = [_Skeleton(ax_p, g["colour"], g["colour"], g["colour"], lw=2.0,
                        left_alpha=LEFT_ALPHA) for g in groups]
    skel_g = [_Skeleton(ax_g, g["colour"], g["colour"], g["colour"], lw=2.0,
                        left_alpha=LEFT_ALPHA) for g in groups]
    for ax in (ax_p, ax_g):
        ax.axhline(0.0, color="#8a6d3b", lw=1.4, alpha=0.6, zorder=0)
        ax.set_aspect("equal"); ax.grid(alpha=0.22)
        ax.set_xlabel("Fore–aft position (m)")
        ax.set_ylim(-0.15, 2.05)
    ax_p.set_xlim(-HALF_P, HALF_P)
    ax_p.set_ylabel("Vertical position (m)")
    ax_p.set_title("Posture, pelvis-aligned", fontsize=11, fontweight="bold")
    ax_g.set_title("Position on the track",
                   fontsize=11, fontweight="bold")

    # The ground camera follows the midpoint of the two groups. Averaged pelvis
    # tracks still carry the within-step surge, so the same smoothing the trial
    # renderer uses is applied here - otherwise the ground grid pulses once per
    # step underneath skeletons that are themselves moving evenly.
    mid_raw = np.mean([g["pelv"][:, 0] for g in groups], axis=0)
    mid_cam = _smooth_track(mid_raw, per_step, win_s=1.0)

    writer = _MP4Writer(fig, out_path, fps)
    try:
        for i in range(n_pts):
            for sp_art, sg_art, g in zip(skel_p, skel_g, groups):
                # LEFT - pelvis at the origin, posture only.
                sp = g["shape"][i].copy()
                sp[:, 2] += g["pelv"][i, 2] - floor
                sp_art.set_frame(sp, h=0, v=2)

                # RIGHT - true pelvis position, so distance is real.
                gp = g["shape"][i] + g["pelv"][i]
                gp[:, 2] -= floor
                sg_art.set_frame(gp, h=0, v=2)

            ax_g.set_xlim(mid_cam[i] - HALF_G, mid_cam[i] + HALF_G)
            header.set_text(
                f"step {i // per_step + 1} of {n_steps}          "
                f"blue {', '.join(f['pids'])}:  {f['pelv'][i,0]:.1f} m at {f['secs'][i]:.2f} s"
                f"      red {', '.join(s['pids'])}:  {s['pelv'][i,0]:.1f} m at {s['secs'][i]:.2f} s")
            writer.add()
    finally:
        writer.close()
    return out_path


# ── Fast group against slow group, the WHOLE run ────────────────────────────

def common_step_count(pids, cap=60):
    """The largest number of steps every athlete in `pids` actually has.

    The step-anchored clock only means anything if "step 20" exists for
    everybody, so the sequence has to stop at the shortest athlete. Over a full
    62 m these six run 29-38 steps, mostly because a slower athlete takes more
    of them to cover the same ground.
    """
    counts = []
    for pid in pids:
        t = clean_for_angles(pid)
        steps, _ = find_first_steps(t["mk"], n_steps=cap)
        counts.append(len(steps))
    return int(min(counts))


def render_representative_full(n=3, n_steps=None, per_step=20, out_path=None,
                               fps=50, dpi=110, track_m=62.5, body_scale=4.0):
    """The whole run, fast group against slow group, two panels.

    The acceleration video stops at step 10, about 16 m in. This one runs the
    trial out to the last step every athlete has — around 29 steps and 60 m — so
    it answers a different question: not "how do they leave the blocks" but
    "where does the gap actually open up".

    LEFT   pelvis-aligned posture, exactly the panel in the acceleration video.
    RIGHT  the ENTIRE track in one fixed view, blocks to finish, no camera
           following anybody. Both groups are on it at once, so the distance
           between them IS the result, and it is printed as a running gap.

    `body_scale` is the one compromise and it is on the panel in words. 62 m of
    track against a 2 m athlete is a 31:1 aspect: drawn true to scale the
    runners come out about forty pixels tall and you cannot see what they are
    doing. Each body is therefore drawn `body_scale` times life size ABOUT ITS
    OWN GROUND CONTACT POINT, which leaves every position — and therefore every
    distance and the gap — exactly where it really is. `body_scale=1.0` gives
    the undistorted view. The right panel's vertical axis is unlabelled because
    of it; vertical measurements belong to the left panel.

    The clock is step-anchored, like the acceleration video and for the same
    reason: it guarantees the three athletes in a group are at the same point of
    the same step, so their average is a body rather than a blur. It also means
    the two groups are NOT at the same instant in time — the slower group takes
    longer to reach any given step — so elapsed seconds are printed for each.
    """
    # 1. Same six athletes as every other group comparison.
    ranked = sorted((clean_for_angles(pid)["peak_vel_ms"], pid)
                    for pid in cohort_pids())
    picks = {"fast": [p for _v, p in ranked[-n:]], "slow": [p for _v, p in ranked[:n]]}

    if n_steps is None:
        n_steps = common_step_count(picks["fast"] + picks["slow"])
    out_path = Path(out_path or
                    MP4_OUT / f"representative_full_top{n}_vs_bottom{n}.mp4")

    # 2. Average each group, shape and position kept separate.
    groups = []
    for key, colour in (("fast", COL_FAST), ("slow", COL_SLOW)):
        seqs = [_accel_sequence(p, n_steps, per_step) for p in picks[key]]
        seqs = [q for q in seqs if q is not None]
        if not seqs:
            raise SystemExit(f"no usable {key} sequences")
        groups.append(dict(
            shape=np.mean([q[0] for q in seqs], axis=0),
            pelv =np.mean([q[1] for q in seqs], axis=0),
            secs =np.mean([q[2] for q in seqs], axis=0),
            pids=picks[key], colour=colour))

    n_pts = groups[0]["shape"].shape[0]
    foot  = [R_HEEL, L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = min(float((g["shape"] + g["pelv"][:, None, :])[:, foot, 2].min())
                for g in groups)

    # 3. Layout, in inches first so both panels can be true to their own aspect.
    #    Fixed rectangles again: the header text changes every frame, and
    #    constrained layout would re-solve and shift the axes under it.
    FW, FH  = 19.2, 5.4
    HALF_P  = 1.3
    Y0, Y1  = -0.15, 2.05
    X0, X1  = -1.5, track_m + 1.5
    G0, G1  = -0.25, 2.15 * body_scale          # right panel, magnified bodies

    fig = _agg_figure((FW, FH), dpi)

    left_w_in  = 4.3
    left_h_in  = left_w_in * (Y1 - Y0) / (2 * HALF_P)
    right_w_in = 13.0
    right_h_in = right_w_in * (G1 - G0) / (X1 - X0)
    bottom_in  = 0.55

    ax_p = fig.add_axes([0.030, bottom_in / FH, left_w_in / FW, left_h_in / FH])
    ax_g = fig.add_axes([0.285,
                         (bottom_in + (left_h_in - right_h_in) / 2) / FH,
                         right_w_in / FW, right_h_in / FH])

    f, s_ = groups[0], groups[1]
    fig.text(0.5, 0.005, DRAW_NOTE, ha="center", va="bottom", fontsize=8, style="italic")
    fig.text(0.5, 0.985, f"Fastest vs slowest runners, whole run (blocks to step {n_steps})",
             ha="center", va="top", fontsize=13, fontweight="bold")
    header = fig.text(0.5, 0.938, "", ha="center", va="top", fontsize=10.5)

    skel_p = [_Skeleton(ax_p, g["colour"], g["colour"], g["colour"], lw=2.0,
                        left_alpha=LEFT_ALPHA) for g in groups]
    skel_g = [_Skeleton(ax_g, g["colour"], g["colour"], g["colour"], lw=1.5,
                        left_alpha=LEFT_ALPHA) for g in groups]
    # At this scale the trail is what carries the story: two lines pulling apart.
    trails = [ax_g.plot([], [], color=g["colour"], lw=1.5, alpha=0.5,
                        solid_capstyle="round")[0] for g in groups]

    for ax in (ax_p, ax_g):
        ax.axhline(0.0, color="#8a6d3b", lw=1.4, alpha=0.6, zorder=0)
        ax.set_aspect("equal")
    ax_p.set_xlim(-HALF_P, HALF_P); ax_p.set_ylim(Y0, Y1)
    ax_p.grid(alpha=0.22)
    ax_p.set_xlabel("Fore–aft position (m)")
    ax_p.set_ylabel("Vertical position (m)")
    ax_p.set_title("Posture, pelvis-aligned", fontsize=11, fontweight="bold")

    ax_g.set_xlim(X0, X1); ax_g.set_ylim(G0, G1)
    ax_g.set_xticks(np.arange(0, track_m + 1, 10))
    ax_g.set_yticks([])                       # bodies are magnified: no vertical scale
    ax_g.grid(axis="x", alpha=0.3)
    ax_g.tick_params(labelsize=9)
    ax_g.set_xlabel("Distance from the blocks (m)")
    ax_g.axvline(0.0, color="#888888", lw=1, ls="--")
    ax_g.axvline(track_m, color="k", lw=1.6)
    ax_g.text(track_m - 0.6, G1, f"{track_m:g} m", fontsize=9,
              fontweight="bold", va="top", ha="right")
    scale_note = ("true scale" if body_scale == 1
                  else f"positions true, bodies drawn {body_scale:g}x for visibility")
    ax_g.set_title(f"Full {track_m:g} m, fixed camera "
                   f"({scale_note})", fontsize=11, fontweight="bold")

    writer = _MP4Writer(fig, out_path, fps)
    try:
        for i in range(n_pts):
            for sp_art, sg_art, trail, g in zip(skel_p, skel_g, trails, groups):
                # LEFT — pelvis at the origin, posture only.
                sp = g["shape"][i].copy()
                sp[:, 2] += g["pelv"][i, 2] - floor
                sp_art.set_frame(sp, h=0, v=2)

                # RIGHT — true pelvis position on a fixed track, body magnified
                # about its own ground point so the feet stay on the floor and
                # the fore-aft position stays exactly where it belongs.
                gp = g["shape"][i] + g["pelv"][i]
                gp[:, 2] -= floor
                gp[:, 0] = g["pelv"][i, 0] + (gp[:, 0] - g["pelv"][i, 0]) * body_scale
                gp[:, 2] *= body_scale
                sg_art.set_frame(gp, h=0, v=2)
                trail.set_data(g["pelv"][: i + 1, 0],
                               (g["pelv"][: i + 1, 2] - floor) * body_scale)

            gap = f["pelv"][i, 0] - s_["pelv"][i, 0]
            header.set_text(
                f"step {i // per_step + 1} of {n_steps}          "
                f"blue {', '.join(f['pids'])}:  {f['pelv'][i,0]:.1f} m at {f['secs'][i]:.2f} s"
                f"      red {', '.join(s_['pids'])}:  {s_['pelv'][i,0]:.1f} m at {s_['secs'][i]:.2f} s"
                f"      gap {gap:+.1f} m")
            writer.add()
    finally:
        writer.close()
    return out_path


# ==========================================================================
# 9 · Run everything, for everyone
# ==========================================================================
#
# Sections 1-6 are where the testing happens, in sprint_analysis.ipynb. So this
# does not pick a representative athlete or a demonstration trial: it renders
# the whole cohort, every time.


def run_all_outputs(n_group=3, n_steps=10, every=1, dpi=100, trials=True, verbose=True):
    """Every figure and every MP4, for every athlete. The whole output stage.

    `trials=False` skips the per-athlete videos, which are the slow part.
    `every=2` halves the frames in those videos for a quick look — the frame
    rate drops with it, so the result is real time but coarser.

    Returns a dict of everything written, keyed by kind.
    """
    written = {"figures": [], "mp4s": []}
    pids_all = cohort_pids()

    def _say(msg):
        if verbose:
            print(msg, flush=True)

    # ── A. The analysis the figures are drawn from: sprint_report's single cohort pass (cached) ─────
    R = _sprint_report()
    c, T = R.load()
    pv = c["athletes"].v_peak_ms.to_dict()                    # smoothed, unscaled peak (the analysis target)
    fpca_acc = run_angle_fpca(dict(zip(sorted(c["accel"]), smooth_angle_curves(
        np.stack([c["accel"][p] for p in sorted(c["accel"])]), penalty=1.0))), n_components=6)

    # ── B. Section 7 — figures ──────────────────────────────────────────────
    # Each plot_* function leaves its figure open for notebooks; closed here as soon as it is on disk.
    _say("\nFigures →")

    def _save(fn, *a, **kw):
        out = fn(*a, **kw)
        plt.close("all")
        written["figures"].append(out)
        return out

    _save(plot_fast_vs_slow, c["smooth"], pv, n=5)
    _save(plot_pc_loading_ranking, c["fpca"], pc=1)
    _save(plot_pc_anatomy, c["raw_top"], c["fpca"], pv, pc=1, k=2.0)
    _save(plot_pc_anatomy, c["raw_acc"], fpca_acc, pv, pc=1, k=2.0,
          title_extra=", acceleration (steps 1-3-5)", fname="pc1_anatomy_acceleration.png")
    _save(plot_velocity_fit, c)

    # Block exit, one panel row per athlete — the phase is athlete-specific, so
    # every athlete gets their own.
    for pid in pids_all:
        try:
            _save(plot_block_exit_steps, pid)
        except Exception as e:
            print(f"  skipped block-exit {pid}: {e}")

    for f in written["figures"][-3:]:
        _say(f"  {f.name}")

    # ── C. Section 8 — MP4s, everyone ───────────────────────────────────────
    _say(f"\nMP4s → {MP4_OUT}")
    if trials:
        for i, pid in enumerate(pids_all, 1):
            try:
                out = render_trial(pid, every=every, dpi=dpi)
                written["mp4s"].append(out)
                _say(f"  [{i:2d}/{len(pids_all)}] {out.name:<28} "
                     f"{out.stat().st_size / 1e6:5.1f} MB")
            except Exception as e:
                print(f"  [{i:2d}/{len(pids_all)}] skipped {pid}: {e}")

    for label, fn in (("top speed",    lambda: render_representative(n=n_group)),
                      ("acceleration", lambda: render_representative_accel(
                          n=n_group, n_steps=n_steps)),
                      ("whole run",    lambda: render_representative_full(
                          n=n_group))):
        try:
            out = fn()
            written["mp4s"].append(out)
            _say(f"  group, {label:<13} {out.name:<40} "
                 f"{out.stat().st_size / 1e6:5.1f} MB")
        except Exception as e:
            print(f"  group {label} failed: {e}")

    _say(f"\n{len(written['figures'])} figures in {FIG_OUT}")
    _say(f"{len(written['mp4s'])} MP4s in {MP4_OUT}")
    return written


# ==========================================================================
# 9 · Analysis figures: one function per figure, shared by sprint_analysis.ipynb and the report
# ==========================================================================
# Each takes the workbook tables `T` (and the cohort cache `c` where curves are needed), returns the
# figure and saves nothing. Titles are short descriptions; the claim and the statistics live in the caption.

STYLE = {"figure.dpi": 110, "savefig.dpi": 200, "font.size": 8, "axes.titlesize": 9,
         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}
SEX_LABEL = {"F": "Women", "M": "Men"}

_NAMES = {
    "v_peak_ms": "Peak velocity (m/s)", "t10_s": "10 m split time (s)", "t20_s": "20 m split time (s)",
    "t30_s": "30 m split time (s)", "height_m": "Height (m)", "leg_m": "Leg length (m)",
    "f_step_hat": "Step frequency, scaled to leg length", "t_contact_hat": "Contact time, scaled to leg length",
    "t_flight_hat": "Flight time, scaled to leg length", "trunk_accel_deg": "Forward trunk lean in the first steps (°)",
    "crp_thighs_deg": "Thigh–thigh relative phase (°)", "thigh_retract_dps": "Thigh retraction velocity (°/s)",
    "knee_ext_dps": "Knee extension velocity (°/s)", "td_dist_m": "Touchdown distance (m)",
    "shank_td_deg": "Shank angle at touchdown (°)", "crp_armleg_deg": "Arm–leg relative phase (°)",
    "bflh_lengthen_Lps": "Hamstring lengthening velocity, late swing (lengths/s)",
    "v_vs_sex_ms": "Peak velocity relative to own sex's mean (m/s)", "t_contact_s": "Contact time (s)",
    "t_flight_s": "Flight time (s)", "f_step_hz": "Step frequency (Hz)", "t_stride_s": "Stride time (s)",
}
_WORD = {"R": "right", "L": "left", "abd": "abduction", "rot": "rotation", "lat": "lateral flexion",
         "axial": "axial rotation", "L5S1": "L5/S1"}


def _cap(s):
    """Sentence case that leaves acronyms alone ("fPC1", "V&B")."""
    return s if s[1:2].isupper() else s[:1].upper() + s[1:]


def label(col):
    """Readable name, with unit, for a workbook column ("knee_lo_ROM" -> "Knee ROM, low-ROM limb (°)")."""
    if col in _NAMES:
        return _NAMES[col]
    if col.startswith("acc_"):                                     # same summary, acceleration steps 1-3-5
        base = label(col[4:])
        return base.replace(" (", ", first steps (", 1) if " (" in base else base + ", first steps"
    if col.startswith("ct_"):
        return f"Contact time, {_CONTACT[col[3:]].lower()} (s)"
    if col.startswith("PC"):
        return f"Principal component {col[2:]} score"
    if col.startswith("share_"):
        return f"{_cap(col[6:-4].replace('_', ' '))}, share of muscle–tendon excursion (%)"
    for key, name in _MUSCLE.items():
        if col.startswith(key + "_r_"):
            return name + {"excursion_pct": " excursion (% of length)",
                           "peak_lengthen_Lps": " peak lengthening velocity (lengths/s)"}[col[len(key) + 3:]]
    if col.endswith(("_ROM", "_mean")):
        *w, stat = col.removeprefix("fx_").split("_")
        side = "".join(_WORD[x] + " " for x in w if x in ("R", "L"))
        body = side + " ".join(_WORD.get(x, x) for x in w if x not in ("R", "L", "hi", "lo"))
        limb = {"hi": ", side that moves more", "lo": ", side that moves less"}
        tail = "".join(limb[x] for x in w if x in limb)
        if stat == "ROM":
            return f"{_cap(body)} range of motion{tail} (°)"
        noun = "" if body.endswith(("lean", "separation", "flexion", "rotation", "abduction", "yaw", "roll")) else " angle"
        return f"Average {_low(body)}{noun}{tail} (°)"
    return _cap(col.replace("_", " "))


def _low(s):
    """Lower-case the first letter unless it starts an acronym ("Hip" -> "hip", "L5/S1" stays)."""
    return s if s[1:2].isupper() or s[1:2].isdigit() else s[:1].lower() + s[1:]


def model_label(m):
    return _MODELS.get(m, _cap(m))


def angle_label(name):
    """ "hip_R" -> "Hip, right"; "trunk_lean" -> "Trunk lean"."""
    return _cap(name.replace("_R", ", right").replace("_L", ", left").replace("_", " "))


_MODELS = {
    "intercept only": "Average only (no predictors)", "sex": "Sex", "height": "Height",
    "knee ROM (hi+lo)": "Knee range of motion, both legs", "knee ROM + height": "Knee range of motion + height",
    "sex + knee ROM + height": "Sex + knee range of motion + height", "22 scalars, ridge": "All 22 angle summaries",
    "fPC1-6 R/L, ridge (fold refit)": "Principal components 1–6 (right/left)",
    "fPC1-6 HI/LO, ridge (fold refit)": "Principal components 1–6 (more/less mobile side)",
    "fPC1-6 R/L unstandardised, ridge": "Principal components 1–6, unscaled angles",
    "trunk lean, accel": "Trunk lean in the first steps", "sex + trunk lean, accel": "Sex + trunk lean in the first steps",
    "22 accel scalars, ridge": "All 22 angle summaries, first steps", "accel fPC1-6, ridge (fold refit)": "Principal components 1–6, first steps",
}
_SHORTCUTS = {
    "top-3 |r| selection on all runners": "Top 3 correlated measures, picked using all runners",
    "best of 1540 3-feature+height models": "Best of 1,540 three-measure + height models",
    "fPCA basis fit on all runners (R/L)": "Principal components fitted on all runners (right/left)",
    "fPCA basis fit on all runners (HI/LO)": "Principal components fitted on all runners (more/less mobile side)",
    "V&B 2023 pipeline: PCA95 + sex, stepwise (15 PCs)": "Vellucci & Beaudette (2023) method, with sex",
    "V&B 2023 pipeline: PCA95 only, stepwise (15 PCs)": "Vellucci & Beaudette (2023) method, kinematics only",
}
_MUSCLE = {"bifemlh": "Biceps femoris long head", "glut_max1": "Gluteus maximus", "iliacus": "Iliacus",
           "rect_fem": "Rectus femoris", "med_gas": "Medial gastrocnemius", "soleus": "Soleus"}
_CONTACT = {"heel_toe_band_s": "Heel–toe height band", "centroid_band_s": "Foot-centre height band",
            "contact_point_s": "Contact-point speed threshold", "ballistic_s": "Flight-time (ballistic) model",
            "contact_point_0.5ms": "Contact-point threshold, 0.5 m/s", "contact_point_1ms": "Contact-point threshold, 1 m/s",
            "ballistic_1mm": "Ballistic model, 1 mm threshold", "ballistic_3mm": "Ballistic model, 3 mm threshold",
            "ballistic_5mm": "Ballistic model, 5 mm threshold", "xsens_speed_s": "Xsens foot-speed contacts"}
_TERMS = {"Intercept": "Intercept", "trunk_deg_w": "Trunk lean, within a runner (m/s per °)",
          "trunk_deg_b": "Trunk lean, between runners (m/s per °)", "f_step_hz_w": "Step frequency, within a runner (m/s per Hz)",
          "f_step_hz_b": "Step frequency, between runners (m/s per Hz)", "sex_m": "Men vs women (m/s)",
          "slope SD (trunk_w)": "Spread of runners' trunk-lean slopes (SD)",
          "r(runner slope, v_peak_ms | sex)": "Runner's slope vs peak velocity, sex held constant (r)",
          "r(runner slope, t10_s | sex)": "Runner's slope vs 10 m split, sex held constant (r)"}
_TARGETS = {"v_peak (pooled)": "Peak velocity, pooled", "v_peak | sex": "Peak velocity, sex held constant",
            "t10 | sex, accel 1-3-5 cycle": "10 m split, sex held constant (acceleration steps)"}
_EVAL_WORDS = [("knee_R", "Right knee"), ("hip_R", "Right hip"), ("ankle_R", "Right ankle"), ("v_peak", "Peak velocity"),
               ("CoM", "centre of mass"), ("HI/LO", "high/low-ROM")]
COLS = {
    "runner_id": "Runner", "sex": "Sex", "n": "n", "r": "r", "p": "p", "r_M": "r, men", "r_F": "r, women",
    "r_part_sex": "Partial r, sex held constant", "r_part": "Partial r, sex held constant", "ci_lo": "95% CI, low",
    "ci_hi": "95% CI, high", "r_part_lo": "95% CI, low", "r_part_hi": "95% CI, high", "p_part": "p (partial)",
    "q_part": "q (false discovery rate)", "dR2_loo_vs_sex": "Gain in LOO R² over sex", "icc_x": "Stride-to-stride ICC",
    "r_part_corrected": "Partial r, corrected for reliability", "variable": "Variable", "b_leg": "Exponent on leg length",
    "hof_b": "Expected exponent (Hof 1996)", "verdict": "Verdict", "target": "Outcome", "angle": "Joint angle",
    "cluster": "Stride window", "peak_r": "Peak r", "mass": "Cluster mass", "variant": "Variant", "pc": "Component",
    "var_pct": "Variance explained (%)", "cum_pct": "Cumulative variance (%)", "top_angle": "Largest loading",
    "top_share_pct": "Largest loading's share (%)", "model": "Model", "loo_r2": "LOO R²", "rmse": "RMSE", "mae": "MAE",
    "perm_p": "Permutation p", "dR2_vs_sex": "Gain over sex", "p_gain_vs_sex": "p (gain over sex)",
    "shortcut": "Shortcut", "reported": "Reported R²", "honest": "Cross-validated R²", "inflation": "Overstatement",
    "null_95": "95th percentile with shuffled speeds", "p_null": "p", "id": "ID", "predicted_sign": "Predicted direction",
    "dR2": "Gain in LOO R² over sex", "p_gain": "p (gain)", "p_holm": "p (Holm)",
    "dR2_min_drop1": "Gain without the most influential runner", "most_influential": "Most influential runner",
    "boot_lo": "Bootstrap 95% CI, low", "boot_hi": "Bootstrap 95% CI, high", "n_implausible": "Implausible values",
    "term": "Term", "coef": "Estimate", "n_steps": "Steps", "n_runners": "Runners", "strides": "Strides pooled",
    "n_strides": "Strides per runner", "median_icc": "Median ICC", "n_icc_gt_075": "Variables with ICC > 0.75",
    "n_icc_lt_050": "Variables with ICC < 0.50", "mean_pct_vpeak": "Mean speed (% of peak)",
    "ik_rms_mm": "IK marker error (mm)", "knee_R_r_osim": "Right knee, r vs OpenSim", "hip_R_r_osim": "Right hip, r vs OpenSim",
    "hip_R_offset_osim_deg": "Right hip offset vs OpenSim (°)", "ik_qa": "IK quality check", "category": "Category",
    "task": "Check", "value": "Value", "tolerance": "Tolerance", "pass_rate": "Pass rate", "detail": "Detail",
    "definition": "Definition", "role": "Role", "mean_ms": "Mean (ms)", "sd_ms": "SD (ms)",
    "pct_runners_80_140ms": "Runners within 80–140 ms (%)", "icc_stride": "Stride-to-stride ICC",
    "frames_per_contact": "Frames per contact",
}


def table(d):
    """A workbook sheet with plain-English column headings and variable names, for display."""
    d = d.copy()
    for col, fn in (("variable", label), ("angle", angle_label), ("top_angle", angle_label), ("model", model_label), ("shortcut", lambda x: _SHORTCUTS.get(x, x)),
                    ("term", lambda x: _TERMS.get(x, x)), ("definition", lambda x: _CONTACT.get(x, x)),
                    ("target", lambda x: _TARGETS.get(x, label(x)))):
        if col in d:
            d[col] = d[col].map(fn)
    if "task" in d:
        for old, new in _EVAL_WORDS:
            d["task"] = d["task"].str.replace(old, new, regex=False)
    if "sex" in d:
        d["sex"] = d["sex"].map(SEX_LABEL).fillna(d["sex"])
    return d.rename(columns=lambda k: COLS.get(k, label(k)))


def _athletes(T):
    return T["athletes"].set_index("runner_id")


def _letters(axes):
    for k, a in zip("ABCDEFGH", axes):
        a.text(-0.08, 1.06, k, transform=a.transAxes, fontweight="bold", fontsize=10, va="bottom")


def _note(a, text):
    a.text(0.03, 0.03, text, transform=a.transAxes, va="bottom", fontsize=7, color="0.3")


def _by_sex(a, d, x, y, fit=False, s=14):
    """Scatter one runner per point, coloured by sex, optionally with each sex's OLS line."""
    for sx in "FM":
        g = d[d.sex == sx].dropna(subset=[x, y])
        a.scatter(g[x], g[y], color=SP.SEX_COL[sx], s=s, label=SEX_LABEL[sx])
        if fit and len(g) > 2:
            b = np.polyfit(g[x], g[y], 1); xx = np.array([g[x].min(), g[x].max()])
            a.plot(xx, np.polyval(b, xx), color=SP.SEX_COL[sx], lw=1)


def _angle_grid(title):
    fig, ax = plt.subplots(3, 5, figsize=(13, 6.6), sharex=True)
    ax = ax.ravel(); [a.axis("off") for a in ax[13:]]
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    for a in ax[10:13]: a.set_xlabel("Stride cycle (%)")
    return fig, ax


def fig_cohort(T, c):
    A = _athletes(T)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for r in A.index:
        vs, fs = c["vel"][r]
        ax[0].plot(np.arange(len(vs)) / fs, vs, color=SP.SEX_COL[A.sex[r]], lw=0.8, alpha=0.7)
    for i, sx in enumerate("FM"):
        d = A[A.sex == sx]
        ax[1].scatter(i + np.random.default_rng(0).uniform(-.12, .12, len(d)), d.v_peak_ms, color=SP.SEX_COL[sx], s=18)
    _by_sex(ax[2], A, "height_m", "v_peak_ms", s=18)
    ax[0].set(xlabel="Time (s)", ylabel="Pelvis speed (m/s)", title="Pelvis speed over the run")
    ax[1].set(xticks=[0, 1], xticklabels=["Women", "Men"], xlim=(-.5, 1.5), ylabel=label("v_peak_ms"), title="Peak velocity by sex")
    ax[2].set(xlabel=label("height_m"), ylabel=label("v_peak_ms"), title="Peak velocity vs height"); ax[2].legend()
    _letters(ax); fig.tight_layout()
    return fig


def fig_bspline(T, c, rid="SB25", angle="knee_R"):
    """How the curves are smoothed: (A) the B-spline basis, (B) one runner's raw vs smoothed knee cycle,
    (C) the same runner's raw vs smoothed pelvis speed."""
    x = np.linspace(0, 100, 101)
    B = SP._bspline_basis(101)
    j = ANGLE_NAMES.index(angle)
    (v_sm, fs), v_raw = c["vel"][rid], c["vel_raw"][rid]
    t = np.arange(len(v_raw)) / fs
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    ax[0].plot(x, B, color="0.55", lw=1)
    ax[0].set(xlabel="Stride cycle (%)", ylabel="Weight", title=f"The {B.shape[1]} cubic B-spline building blocks")
    ax[1].plot(x, c["cycles"][rid][:, j], "o", color="0.6", ms=2.5, label="Measured (60 Hz, time-normalised)")
    ax[1].plot(x, c["smooth"][rid][:, j], color=INK, lw=1.6, label="Penalised B-spline fit")
    side, joint = ("Right", angle[:-2]) if angle.endswith("_R") else ("Left", angle[:-2])
    ax[1].set(xlabel="Stride cycle (%)", ylabel=f"{_cap(joint)} angle (°)", title=f"{rid}: {side.lower()} {joint} over one top-speed stride")
    ax[1].legend(loc="upper left", fontsize=7)
    ax[2].plot(t, v_raw, color="0.6", lw=.8, label="Measured (differentiated position)")
    ax[2].plot(t, v_sm, color=INK, lw=1.6, label="Penalised B-spline fit")
    ax[2].set(xlabel="Time (s)", ylabel="Pelvis speed (m/s)", title=f"{rid}: pelvis speed over the run")
    ax[2].legend(loc="lower right", fontsize=7)
    _letters(ax); fig.tight_layout()
    return fig


def fig_fpca(T, c, pc=1, k=2.0):
    """What one functional principal component means: (A, B) the two angles it loads on most, drawn for a runner
    k SD below and above the mean score; (C) every angle's loading; (D) each runner's score against peak velocity."""
    f, A = c["fpca"], _athletes(T)
    L, sd, s = f["loadings"][pc - 1], f["angle_sd"], f["scores"][:, pc - 1]
    imp = np.linalg.norm(L, axis=0); top = np.argsort(-imp)[:2]
    x = np.linspace(0, 100, f["n_points"])
    curve = lambda kk, j: (f["mean_curve"][:, j] + kk * s.std() * L[:, j]) * sd[j]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    for a, j in zip(ax[:2], top):
        a.plot(x, curve(0, j), color="0.6", lw=1.2, label="Mean runner")
        a.plot(x, curve(-k, j), color=PC_MINUS, lw=1.6, label=f"Low score (−{k:g} SD)")
        a.plot(x, curve(k, j), color=PC_PLUS, lw=1.6, label=f"High score (+{k:g} SD)")
        a.set(xlabel="Stride cycle (%)", ylabel="Angle (°)", title=angle_label(f["names"][j]))
    ax[1].legend(loc="lower right", fontsize=7)
    o = np.argsort(imp)
    ax[2].barh([angle_label(f["names"][i]) for i in o], imp[o], color="0.6")
    ax[2].set(xlabel="Loading (size across the stride)", title=f"Which angles PC{pc} uses")
    sc = pd.Series(s, index=f["pids"]).reindex(A.index)
    d = A.assign(score=sc)
    _by_sex(ax[3], d, "score", "v_peak_ms", s=16)
    ax[3].set(xlabel=f"PC{pc} score", ylabel=label("v_peak_ms"), title=f"PC{pc} score vs peak velocity"); ax[3].legend(fontsize=7)
    fig.suptitle(f"Functional PC{pc}: {f['explained_var'][pc - 1]:.0f}% of the variation in joint-angle curves", x=0.01, ha="left", fontsize=10)
    _letters(ax); fig.tight_layout()
    return fig


def fig_pooled_vs_adjusted(T, n=8):
    """Pooled and sex-adjusted correlation with peak velocity for the n variables with the largest pooled |r|."""
    s = T["screen"].reindex(T["screen"].r.abs().sort_values().index).tail(n)
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * n + 1.2))
    ax.hlines(y, s.r, s.r_part_sex, color="0.75")
    ax.scatter(s.r, y, color="0.6", s=40, label="Pooled", zorder=3)
    ax.scatter(s.r_part_sex, y, color=COL_FAST, s=40, label="Sex-adjusted (partial r)", zorder=3)
    ax.axvline(0, color="k", lw=.5)
    ax.set(yticks=y, yticklabels=[label(v).removesuffix(" (°)").removesuffix(" (m)") for v in s.variable],
           xlabel="Correlation with peak velocity", title="Correlation with peak velocity, pooled and sex-adjusted")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0)); fig.tight_layout()
    return fig


def fig_screen(T, n=8):
    """The n variables with the largest sex-adjusted correlation, each against peak velocity with within-sex fits."""
    A = _athletes(T)
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.6)); ax = ax.ravel()
    for a, (_, row) in zip(ax, T["screen"].head(n).iterrows()):
        _by_sex(a, A, row.variable, "v_peak_ms", fit=True)
        a.set(xlabel=label(row.variable), title=f"r = {row.r:+.2f}, partial r = {row.r_part_sex:+.2f}")
    for a in ax[::4]: a.set_ylabel(label("v_peak_ms"))
    ax[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Top eight variables by sex-adjusted correlation with peak velocity", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def fig_stride_correlation(T, c):
    """Pointwise correlation of each joint angle with peak velocity, pooled and sex-adjusted, with SPM clusters."""
    import scipy.stats as st
    R = _sprint_report()
    A = _athletes(T)
    S = np.stack([c["smooth"][r] for r in A.index])                    # (runners, 101, angles)
    v, sex = A.v_peak_ms.to_numpy(), (A.sex == "M").to_numpy(float)
    tc = st.t.ppf(.975, len(v) - 2); rcrit = tc / np.sqrt(len(v) - 2 + tc ** 2)
    x = np.arange(S.shape[1])
    fig, ax = _angle_grid("Correlation between joint angle and peak velocity across the stride")
    for j, a in enumerate(ax[:13]):
        a.plot(x, SP._tstat_curve(S[:, :, j], v)[1], color="0.6", lw=1)
        a.plot(x, SP._tstat_curve(R._resid(S[:, :, j], sex), R._resid(v, sex))[1], color=COL_FAST, lw=1.4)
        a.axhline(0, color="k", lw=.5); [a.axhline(k * rcrit, color="k", lw=.5, ls="--") for k in (-1, 1)]
        for cl in T["_spm_detail"].get(ANGLE_NAMES[j], {}).get("clusters", []):
            if cl["p"] < .10:
                a.axvspan(cl["start_pct"], cl["end_pct"], color="g" if cl["p"] < .05 else "orange", alpha=.15)
        a.set(title=angle_label(ANGLE_NAMES[j]), ylim=(-.8, .8))
    for a in ax[::5]: a.set_ylabel("r")
    ax[13].legend(handles=[plt.Line2D([], [], color="0.6", label="Pooled"), plt.Line2D([], [], color=COL_FAST, label="Sex-adjusted"),
                           plt.Line2D([], [], color="k", ls="--", lw=.5, label="p = .05, uncorrected"),
                           plt.Rectangle((0, 0), 1, 1, color="orange", alpha=.15, label="SPM cluster, p < .10")], loc="center")
    fig.tight_layout()
    return fig


def fig_fast_vs_slow_within_sex(T, c, n=5):
    """Mean angle curves of the n fastest and n slowest runners relative to their own sex's mean."""
    A = _athletes(T)
    S = np.stack([c["smooth"][r] for r in A.index])
    order = A.v_vs_sex_ms.sort_values()
    fast, slow = [A.index.get_loc(r) for r in order.index[-n:]], [A.index.get_loc(r) for r in order.index[:n]]
    x = np.arange(S.shape[1])
    fig, ax = _angle_grid(f"Mean joint angles, {n} fastest vs {n} slowest runners within sex")
    for j, a in enumerate(ax[:13]):
        m, sd = S[fast, :, j].mean(0), S[fast, :, j].std(0)
        a.fill_between(x, m - sd, m + sd, color=COL_FAST, alpha=.12, lw=0)
        a.plot(x, m, color=COL_FAST, lw=1.4, label=f"Fastest {n} (±1 SD)")
        a.plot(x, S[slow, :, j].mean(0), color=COL_SLOW, lw=1.4, label=f"Slowest {n}")
        a.set_title(angle_label(ANGLE_NAMES[j]))
    for a in ax[::5]: a.set_ylabel("Angle (°)")
    ax[13].legend(*ax[0].get_legend_handles_labels(), loc="center")
    fig.tight_layout()
    return fig


def fig_models(T):
    """(A) Leave-one-out R² of each pre-specified model; (B) R² as reported by common shortcuts vs cross-validated."""
    m, L = T["models"], T["leakage"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax[0].barh([model_label(k) for k in m.model[::-1]], m.loo_r2[::-1], color="0.6")
    sx = m.set_index("model").loc["sex", "loo_r2"]
    ax[0].axvline(sx, color="k", ls="--", lw=1)
    ax[0].text(sx + 0.02, 0.98, "Sex only", transform=ax[0].get_xaxis_transform(), rotation=90, va="top", fontsize=7)
    ax[0].axvline(0, color="k", lw=.5)
    ax[0].set(xlabel="Leave-one-out R²", title="Leave-one-out R² by model")
    yy = np.arange(len(L))[::-1]
    ax[1].hlines(yy, L.honest, L.reported, color="0.75")
    ax[1].scatter(L.reported, yy, color="0.6", label="Reported", zorder=3)
    ax[1].scatter(L.honest, yy, color=COL_FAST, label="Cross-validated", zorder=3); ax[1].axvline(0, color="k", lw=.5)
    ax[1].set(yticks=yy, yticklabels=[_SHORTCUTS.get(k, k) for k in L.shortcut], xlabel="R²", title="Reported vs cross-validated R²")
    ax[1].legend(loc="lower right")
    _letters(ax); fig.tight_layout()
    return fig


def fig_splits(T):
    """(A) 20 m split vs peak velocity; (B) 20 m split vs acceleration trunk lean; (C) LOO R² of sex vs sex + trunk lean."""
    A = _athletes(T)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    _by_sex(ax[0], A, "v_peak_ms", "t20_s", s=16); _note(ax[0], f"r = {A.v_peak_ms.corr(A.t20_s):+.2f}")
    _by_sex(ax[1], A, "trunk_accel_deg", "t20_s", fit=True, s=16)
    acc = T["accel"].pivot(index="target", columns="model", values="loo_r2")[["sex", "sex + trunk lean, accel"]]
    acc.index = [f"{k[1:3]} m" for k in acc.index]
    acc.columns = ["Sex", "Sex + trunk lean"]
    acc.plot.bar(ax=ax[2], color=["0.6", COL_FAST], rot=0, width=.7)
    ax[0].set(xlabel=label("v_peak_ms"), ylabel=label("t20_s"), title="20 m split time vs peak velocity"); ax[0].legend(loc="upper right")
    ax[1].set(xlabel=label("trunk_accel_deg"), ylabel=label("t20_s"), title="20 m split time vs trunk lean")
    ax[2].set(xlabel="Split", ylabel="Leave-one-out R²", ylim=(0.5, 0.8), title="Leave-one-out R² by split"); ax[2].legend(fontsize=7)
    _letters(ax); fig.tight_layout()
    return fig


def fig_prespecified(T):
    """Each pre-registered variable (and earlier lead) against its outcome, with within-sex fits."""
    A, P = _athletes(T), T["prespecified"].drop_duplicates("variable")
    fig, ax = plt.subplots(2, 4, figsize=(11, 5.2)); ax = ax.ravel()
    for a, (_, row) in zip(ax, P.iterrows()):
        _by_sex(a, A, row.variable, row.target, fit=True, s=10)
        a.set(xlabel=label(row.variable), ylabel=label(row.target), title=f"{row.id}: ΔLOO R² = {row.dR2:+.3f}, p = {row.p_gain:.3f}")
    for a in ax[len(P):]: a.axis("off")
    ax[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Pre-registered variables vs outcome, with within-sex fits", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def fig_mtu(T):
    """(A) Each muscle group's share of muscle–tendon excursion; (B) hamstring lengthening velocity vs peak velocity."""
    A, mt = _athletes(T), T["muscle_mtu"].set_index("runner_id")
    sh = mt[[k for k in mt.columns if k.startswith("share_")]].dropna()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax[0].boxplot(sh.to_numpy(), tick_labels=[_cap(k[6:-4].replace("_", " ")) for k in sh.columns], medianprops={"color": "k"})
    ax[0].tick_params(axis="x", rotation=30)
    ax[0].set(ylabel="Share of muscle–tendon excursion (%)", title=f"Muscle–tendon excursion by group, top-speed stride (n = {len(sh)})")
    _by_sex(ax[1], A, "bflh_lengthen_Lps", "v_peak_ms", s=16)
    ax[1].set(xlabel=label("bflh_lengthen_Lps"), ylabel=label("v_peak_ms"), title="Hamstring lengthening vs peak velocity"); ax[1].legend()
    _letters(ax); fig.tight_layout()
    return fig


def fig_agreement(T):
    """Pipeline vs Xsens MVN: Bland–Altman for peak velocity and stride time, joint-angle offsets, contact counts."""
    A = _athletes(T)
    X = T["xsens"].set_index("runner_id").loc[A.index]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for a, (p, q, name, u) in zip(ax[:2], [(A.v_peak_ms, X.v_peak_xsens_ms, "Peak velocity", "m/s"),
                                           (X.t_stride_pipe_s, X.t_stride_xsens_s, "Stride time", "s")]):
        m, d = (p + q) / 2, p - q
        a.scatter(m, d, s=14, c=[SP.SEX_COL[sx] for sx in A.sex]); a.axhline(d.mean(), color="k", lw=1)
        [a.axhline(d.mean() + k * 1.96 * d.std(), color="k", lw=.6, ls="--") for k in (-1, 1)]
        a.set(xlabel=f"Mean of methods ({u})", ylabel=f"Pipeline − Xsens ({u})", title=f"{name}, pipeline vs Xsens")
    for k, jn in enumerate(["knee_R", "hip_R", "ankle_R"]):
        ax[2].scatter(np.full(len(X), k) + np.random.default_rng(k).uniform(-.15, .15, len(X)), X[f"{jn}_offset_deg"], s=12, color="0.4")
    ax[2].axhline(0, color="k", lw=.5)
    ax[2].set(xticks=range(3), xticklabels=["Knee", "Hip", "Ankle"], ylabel="Mean offset (°)", title="Joint-angle offset from Xsens")
    dn = X.n_contacts_R_pipe - X.n_contacts_R_xsens
    ax[3].hist(dn, bins=np.arange(dn.min() - .5, dn.max() + 1.5), color="0.5"); ax[3].axvspan(-1.5, 1.5, color="g", alpha=.1)
    ax[3].set(xlabel="Right-foot contacts, pipeline − Xsens", ylabel="Runners", title="Contact count difference")
    _letters(ax); fig.tight_layout()
    return fig


# ==========================================================================
# 10 · The report (StealIQ format, short): every number read from the workbook
# ==========================================================================

REPORT = Path(__file__).resolve().parent / "Sprint_60m_Report.pdf"

# Evidence check: each claim against the 390-draft sources (quotes verbatim from the abstracts, fetched by DOI
# 2026-10-11), † = method/instrument reference outside the 390 list. Verdicts: consistent / qualifies /
# hypothesis-generating / not testable here / contradicts / no support in sources.
EVIDENCE = [
    ("E1", "Sex alone explains most between-runner top speed", "models: sex LOO R² {sex:.3f}",
     "Vellucci & Beaudette (2023)", "“We recruited 40 healthy university age sprint-based athletes” (sex composition not used as a baseline)",
     "no support in sources (new); qualifies V&B"),
    ("E2", "Whole-body PCA + stepwise fit is reproduced, but carried by sex", "leakage: in-sample {vb_in:.3f} → LOO {vb_loo:.3f}; kinematics only {vb_kin:+.3f}",
     "Vellucci & Beaudette (2023)", "“Both the kinematic (R2 = 0.795) and sEMG data (R2 = 0.586) produced significant multivariate linear regression models.”",
     "qualifies"),
    ("E3", "Contralateral arm–leg timing relates to speed", "H5: ΔLOO {h5_d:+.3f}, p {h5_p:.3f}, Holm {h5_h:.2f}",
     "Vellucci & Beaudette (2023)", "“…the timing of the contralateral arm and leg swing…”", "hypothesis-generating (direction consistent)"),
    ("E4", "Top-speed contact ≈ 0.1 s; heel/toe definition plausible, foot-centre too short", "contact_defs: heel/toe {ct_ht:.0f} ms, foot-centre {ct_cen:.0f} ms",
     "Weyand et al. (2010); Čoh et al. (2018)", "“…forward running (T(c) = … 0.110 ± 0.005 s)”; Bolt: “contact and flight times were 86 and 145 ms”",
     "consistent (heel/toe); contradicts foot-centre"),
    ("E5", "Higher step frequency / shorter contact go with speed", "L4 normalised step frequency ΔLOO {l4_d:+.3f} (not robust); contact not robust across definitions",
     "Morin et al. (2012)", "“Step frequency, contact and swing time were significantly correlated to acceleration and 100-m performance (positively for the former, negatively for the two latter)”",
     "consistent direction; contact not measurable at 60 Hz"),
    ("E6", "Within sex, body size does not scale performance (targets kept absolute)", "size: b(v_peak) {b_v:+.2f} [{b_v_lo:+.2f}, {b_v_hi:+.2f}]",
     "Morin et al. (2012); Hof (1996)†", "“…lower-limb-to-height ratio showed no significant correlation with 100-m performance.”", "consistent"),
    ("E7", "More forward trunk lean out of the blocks goes with faster splits", "L1: ΔLOO {l1_d:+.3f}, within-sex p {l1_p:.4f}; SPM trunk cluster p {spm_tr:.4f}",
     "Debaere et al. (2013)", "“…the sprinter actively searches a more forward leaning position to maximize horizontal velocity.”",
     "consistent; hypothesis-generating (one runner carries half)"),
    ("E8", "Within a run, the trunk rises and steps quicken as speed builds", "steps_mixed: trunk_w {mm_tr:+.3f} m/s per °, step-frequency_w {mm_f:+.2f} m/s per Hz, {n_steps} steps",
     "von Lieres und Wilkau et al. (2020)", "“…step-to-step changes in shank and trunk angles might provide an appropriate measure to detect sprint phases…”", "consistent"),
    ("E9", "Hip extension / thigh retraction velocity drives top speed", "H1: ΔLOO {h1_d:+.3f}, p {h1_p:.2f}",
     "Colyer et al. (2018); Schache et al. (2011)", "Colyer: force-waveform study (“…a higher antero-posterior component of force…”), no hip-velocity measure; Schache: “…hip extensor and knee flexor muscles during terminal swing demonstrated the most dramatic increase in biomechanical load…”",
     "not supported here; 390 draft's Colyer citation for hip extension velocity is not backed by its abstract"),
    ("E10", "Hamstring demand peaks in late swing at top speed", "H6 (MTU, n {h6_n}): ΔLOO {h6_d:+.3f}, p {h6_p:.2f}",
     "Schache et al. (2011); Higashihara et al. (2018)", "Higashihara: “During the late stance and terminal mid-swing of maximum-speed sprint, the knee was more extended and a higher knee flexion moment was observed…”",
     "rationale consistent (kinetic/EMG); kinematic stretch speed does not separate runners"),
    ("E11", "Shank angle at touchdown matters early in acceleration", "H4: ΔLOO {h4_d:+.3f}, p {h4_p:.3f}, Holm {h4_h:.3f}",
     "Debaere et al. (2015); von Lieres und Wilkau et al. (2020)", "Debaere 2015: “During the first stance, where the athlete runs in a forward bending position, the knee contributes primarily to body lift…”",
     "hypothesis-generating"),
    ("E12", "Foot lands ~0.3–0.4 m ahead of the CoM at top-speed touchdown (assistant's claim)", "H3: 17/30 implausible at 60 Hz",
     "—", "No source in the 390 list.", "no support in sources; not measurable at 60 Hz"),
    ("E13", "Force orientation / horizontal force determines performance", "no GRF recorded",
     "Morin et al. (2011); Rabita et al. (2015); Colyer et al. (2018)", "Morin 2011: “…the orientation of the total force applied onto the supporting ground during sprint acceleration is more important to performance than its amount.”",
     "not testable here"),
    ("E14", "Muscle activations can be inferred from these data", "OpenSim used for MTU lengths only",
     "Debaere et al. (2015); Higashihara et al. (2018)", "Debaere 2015 needed “two force plates” and “muscle-driven simulations”; Higashihara used EMG.",
     "contradicts (activation needs force/EMG)"),
    ("E15", "Non-linear ML captures speed better than linear models (390 purpose)", "honest LOO: no learner beats sex; leakage ledger",
     "Rabita et al. (2015)", "Force–velocity relations “…were well described by significant linear … and quadratic … models”", "qualifies: at n = 30 non-linear gains are untestable"),
    ("E16", "390 Methods: Ridge/Lasso/Elastic Net called “support vector regression”; K-fold without nested selection", "leakage: search 0.403 → {srch:+.3f}",
     "Vellucci & Beaudette (2023) design", "Methods text, 390 draft ¶ “Analysis in Regression”.", "contradicts (mislabelled methods; selection leak)"),
    ("E17", "Contact timing needs > 60 Hz (assistant's claim)", "contact range across definitions {ct_lo:.0f}–{ct_hi:.0f} ms",
     "Weyand et al. (2010); Xsens MVN whitepaper†", "Weyand used a “custom, high-speed force treadmill”; Xsens: MVN Link (240 Hz) recommended “for high dynamic movements including frequent interactions with the floor”.",
     "consistent"),
]


def _evidence_numbers(T):
    m, P, L = T["models"].set_index("model"), T["prespecified"].set_index("id"), T["leakage"]
    cd, sz, mm = T["contact_defs"].set_index("definition"), T["size"].set_index("variable"), T["steps_mixed"]
    spm = T["spm"]; tr = spm[(spm.target.str.startswith("t10")) & (spm.angle == "trunk_lean")].p.min()
    g = lambda i, c: P.loc[i, c] if i in P.index else np.nan
    main = cd.loc[["heel_toe_band_s", "centroid_band_s", "contact_point_s", "ballistic_s"], "mean_ms"]
    return dict(sex=m.loc["sex", "loo_r2"], vb_in=L.reported[4], vb_loo=L.honest[4], vb_kin=L.honest[5],
                h5_d=g("H5", "dR2"), h5_p=g("H5", "p_gain"), h5_h=g("H5", "p_holm"),
                ct_ht=cd.loc["heel_toe_band_s", "mean_ms"], ct_cen=cd.loc["centroid_band_s", "mean_ms"],
                l4_d=g("L4", "dR2"), b_v=sz.loc["v_peak_ms", "b_leg"], b_v_lo=sz.loc["v_peak_ms", "ci_lo"], b_v_hi=sz.loc["v_peak_ms", "ci_hi"],
                l1_d=g("L1", "dR2"), l1_p=g("L1", "p_gain"), spm_tr=tr,
                mm_tr=mm.set_index("term").coef["trunk_deg_w"], mm_f=mm.set_index("term").coef["f_step_hz_w"], n_steps=int(mm.n_steps.iloc[0]),
                h1_d=g("H1", "dR2"), h1_p=g("H1", "p_gain"), h6_n=int(g("H6", "n")), h6_d=g("H6", "dR2"), h6_p=g("H6", "p_gain"),
                h4_d=g("H4", "dR2"), h4_p=g("H4", "p_gain"), h4_h=g("H4", "p_holm"), srch=L.honest[1],
                ct_lo=main.min(), ct_hi=main.max())


def build_report():
    """Short report in the StealIQ paper format -> Sprint_60m_Report.pdf (+ the `evidence` sheet in the workbook)."""
    R = _sprint_report()
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"         # DejaVu has Δ, č, ≈, → (reportlab's base-14 fonts do not)
    for name, f in (("Serif", "DejaVuSerif.ttf"), ("Serif-Bold", "DejaVuSerif-Bold.ttf"),
                    ("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(ttf / f)))
    c, T = R.load()
    A = T["athletes"].set_index("runner_id")
    n = _evidence_numbers(T)
    ev = pd.DataFrame([dict(id=i, claim=cl, our_evidence=o.format(**n), sources=src, source_says=q, verdict=v)
                       for i, cl, o, src, q, v in EVIDENCE])
    with pd.ExcelWriter(R.XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        ev.to_excel(w, sheet_name="evidence", index=False)

    INK, MUTED = colors.HexColor("#111111"), colors.HexColor("#55544f")
    TITLE = ParagraphStyle("T", fontName="Sans-Bold", fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=4)
    SUB = ParagraphStyle("S", fontName="Sans", fontSize=10.5, leading=13, textColor=MUTED, alignment=TA_CENTER, spaceAfter=12)
    H1 = ParagraphStyle("H1", fontName="Sans-Bold", fontSize=12, leading=15, spaceBefore=10, spaceAfter=4, keepWithNext=1)
    H2 = ParagraphStyle("H2", fontName="Sans-Bold", fontSize=10, leading=13, spaceBefore=6, spaceAfter=2, keepWithNext=1)
    P_ = ParagraphStyle("P", fontName="Serif", fontSize=10.2, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=5)
    ABS = ParagraphStyle("A", parent=P_, fontSize=9.6, leading=12.6, leftIndent=22, rightIndent=22)
    CAP = ParagraphStyle("C", fontName="Sans", fontSize=8, leading=10.2, spaceBefore=3, spaceAfter=10)
    CELL = ParagraphStyle("Ce", fontName="Sans", fontSize=7.3, leading=8.9)
    CELLB = ParagraphStyle("Cb", parent=CELL, fontName="Sans-Bold")
    NOTE = ParagraphStyle("N", fontName="Sans", fontSize=7.2, leading=9, textColor=MUTED, spaceAfter=9)
    REF = ParagraphStyle("R", parent=P_, fontSize=8.8, leading=11, leftIndent=14, firstLineIndent=-14, alignment=0)
    para = lambda t, st=P_: Paragraph(t, st)

    def table(rows, widths, caption, note=None):
        data = [[Paragraph(str(x), CELLB if i == 0 else CELL) for x in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
        t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.9, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
                               ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 1.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)]))
        return [para(f"<b>Table {caption[0]}.</b> {caption[1]}", CAP), t, para(note, NOTE) if note else Spacer(1, 8)]

    def figure(path, k, cap, width=6.5):
        w, h = ImageReader(str(path)).getSize()
        return KeepTogether([Image(str(path), width=width * inch, height=width * inch * h / w), para(f"<b>Figure {k}.</b> {cap}", CAP)])

    # ── figures (drawn here, from the workbook) ──
    fdir = cohort_dir(FIG_OUT, "Report")
    m, L, P = T["models"], T["leakage"], T["prespecified"]
    with plt.rc_context(STYLE):
        f1, f2 = fdir / "fig1_validation.png", fdir / "fig2_prespecified.png"
        for fn, out in ((fig_models, f1), (fig_prespecified, f2)):
            fig = fn(T); fig.savefig(out); plt.close(fig)

    St = c["steps"].merge(A[["sex"]], left_on="runner_id", right_index=True)
    St["trunk_w"] = St.trunk_deg - St.groupby("runner_id").trunk_deg.transform("mean")
    St["speed_w"] = St.speed_ms - St.groupby("runner_id").speed_ms.transform("mean")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].scatter(St.trunk_w, St.speed_w, s=3, alpha=.35, color=COL_FAST)
    k_ = n["mm_tr"]; xx = np.array([St.trunk_w.min(), St.trunk_w.max()]); ax[0].plot(xx, k_ * xx, color="k", lw=1.2)
    ax[0].set(xlabel="Trunk lean, deviation from runner mean (°)", ylabel="Step speed deviation (m/s)",
              title="Step speed vs trunk lean, within runners")
    mt = T["muscle_mtu"].set_index("runner_id"); sh = mt[[k for k in mt.columns if k.startswith("share_")]].dropna()
    ax[1].boxplot(sh.to_numpy(), tick_labels=[_cap(k[6:-4].replace("_", " ")) for k in sh.columns], medianprops={"color": "k"}); ax[1].tick_params(axis="x", labelsize=7, rotation=30)
    ax[1].set(ylabel="Share of muscle–tendon excursion (%)", title="Muscle–tendon excursion by group, top-speed stride")
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    _letters(ax)
    fig.tight_layout(); f3 = fdir / "fig3_within_mtu.png"; fig.savefig(f3, dpi=200); plt.close(fig)

    # ── text ──
    men, wom = A[A.sex == "M"].v_peak_ms, A[A.sex == "F"].v_peak_ms
    e = T["eval"].set_index("task")
    vrow = e.loc["v_peak vs Xsens CoM peak"]; knee = e.loc["knee_R cycle vs Xsens ISB angle (r)"]
    cd = T["contact_defs"].set_index("definition")
    pre = P.set_index("id"); mm = T["steps_mixed"].set_index("term")
    story = [para("Sprint Technique and Speed: What 30 Runners Can Show", TITLE),
             para("An honest-validation audit of a 60 m Xsens dataset, with every claim checked against its sources", SUB),
             para(f"<b>Abstract.</b> Thirty university sprinters (15 men, 15 women) ran 60 m in a 17-sensor Xsens MVN suit recorded at "
                  f"60 Hz. We asked which kinematics predict top speed and acceleration once the comparison is honest: every model is "
                  f"scored on runners it never saw, against sex alone as the baseline, and every new variable was pre-registered before it "
                  f"was computed. Sex alone predicts top speed at leave-one-out R² {n['sex']:.2f}. A published whole-body PCA pipeline "
                  f"re-run on this cohort reaches R² {n['vb_in']:.2f} in-sample but carries no kinematic information beyond sex "
                  f"(kinematics only, held out: {n['vb_kin']:+.2f}), and an exhaustive model search that reports 0.40 scores "
                  f"{n['srch']:+.2f} when the search is repeated inside each fold. The pipeline agrees with Xsens's own model "
                  f"(knee r ≈ {knee['value']:.2f}; peak speed within ±10 % for all runners). Of six pre-registered new variables, none "
                  f"survives correction and robustness checks; trunk lean out of the blocks and leg-length-normalised step frequency "
                  f"remain hypotheses that one runner largely carries. The clearest kinematic result is within runners: across "
                  f"{n['n_steps']:,} steps, each runner's trunk rises and steps quicken as speed builds. Seventeen claims, including "
                  f"this project's earlier ones, are checked against the sources of its manuscript draft.", ABS),
             para("1. Introduction", H1),
             para("Sprint speed is often explained through stance mechanics, ground force and its orientation (Weyand et al., 2010; Morin "
                  "et al., 2011, 2012), and increasingly through whole-body coordination (Vellucci &amp; Beaudette, 2023). This project set "
                  "out to predict top speed and acceleration from whole-body kinematics with machine learning. With 30 runners, a "
                  "model that fits well in-sample can say little about a new runner, so the questions here are narrower and testable: "
                  "what predicts an unseen runner beyond sex, what the measurements can and cannot resolve at 60 Hz, and how far the "
                  "literature the draft relies on supports each claim."),
             para("2. Data and measurements", H1),
             para(f"Thirty-one sprinters were recorded; one (SB17) was excluded for tracking loss, leaving 15 men (peak speed "
                  f"{men.mean():.2f} ± {men.std():.2f} m/s) and 15 women ({wom.mean():.2f} ± {wom.std():.2f} m/s). Peak speed is the "
                  f"maximum of a penalised B-spline fitted to pelvis speed; split times are the time from first movement until the pelvis "
                  f"has travelled 10, 20 and 30 m. Thirteen sagittal joint angles are averaged over the five strides nearest peak speed "
                  f"(time-normalised to 101 points). Contact time uses the heel/toe-landmark band (mean {n['ct_ht']:.0f} ms), chosen "
                  f"before any result on plausibility and stride-to-stride reliability; three alternatives span {n['ct_lo']:.0f}–"
                  f"{n['ct_hi']:.0f} ms, because at 60 Hz a contact lasts only five to six frames. Muscle–tendon lengths come from "
                  f"OpenSim inverse kinematics on gait2392 scaled to stature (kinematics only: no body mass, force or EMG was recorded, "
                  f"so muscle activation is not estimable)."),
             para("3. Methods", H1),
             para("Every model is scored by leave-one-out cross-validation with all preprocessing and selection refit inside each fold. "
                  "Each result is reported as its gain over sex alone (ΔLOO). New variables were written into the project README "
                  "with their direction before computation (H1–H6), tested by a within-sex permutation test of the gain (5,000 "
                  "shuffles) with Holm correction across H1–H5 (Holm, 1979†), and accepted only if the gain survives removing its most "
                  "influential runner (≥ +0.03) and its bootstrap 95 % interval excludes zero. Variables with physiologically impossible "
                  "values in more than 10 % of runners are reported as not measurable. Within-runner effects use a linear mixed model "
                  "of step speed with runner random intercepts and slopes. Measurement validity is checked against Xsens MVN's own "
                  "exports (same sensors, independent processing), following the benchmark design of BiomechAgent (Cotton &amp; Leonard, "
                  "2026†): fixed tolerances, a reference the pipeline never reads, and per-task pass rates."),
             para("4. Findings", H1), para("4.1 Honest validation: sex is the baseline to beat", H2),
             para(f"Sex alone predicts top speed at LOO R² {n['sex']:.3f}. Knee range of motion with stature reaches "
                  f"{m.set_index('model').loc['knee ROM + height', 'loo_r2']:.3f} but adds nothing once sex is included "
                  f"({m.set_index('model').loc['sex + knee ROM + height', 'loo_r2']:.3f}). No functional-PCA model, standardised or not, "
                  f"beats the intercept (Figure 1)."),
             figure(f1, 1, "Leave-one-out R² by model and reported vs cross-validated R². (A) Leave-one-out R² of each pre-specified model for peak velocity; the dashed line marks sex alone. (B) R² as reported when a selection step uses all runners (grey) against the same step refit inside each cross-validation fold (blue)."),
             para("4.2 Measurement validity", H2),
             para(f"Against Xsens MVN's own model, peak speed agrees within ±10 % for {vrow['pass_rate']:.0%} of runners ({vrow['detail']}), "
                  f"split times for all runners (r = 0.999) and the knee cycle at median r = {knee['value']:.3f}. The hip differs by a "
                  f"definition offset (thigh against trunk here, against pelvis in Xsens). Five of 30 OpenSim fits failed quality checks "
                  f"and were excluded from the muscle–tendon results."),
             para("4.3 Pre-registered hypotheses", H2),
             *table([["ID", "Variable → target", "partial r [95% CI]", "ΔLOO", "p (Holm)", "ΔLOO w/o top runner", "Verdict"]] +
                    [[i, f"{r_.variable} → {r_.target}", f"{r_.r_part:+.2f} [{r_.r_part_lo:+.2f}, {r_.r_part_hi:+.2f}]", f"{r_.dR2:+.3f}",
                      f"{r_.p_gain:.3f}" + (f" ({r_.p_holm:.3f})" if pd.notna(r_.p_holm) else ""), f"{r_.dR2_min_drop1:+.3f}", r_.verdict]
                     for i, r_ in pre.iterrows()],
                    [0.35, 1.55, 1.05, 0.5, 0.75, 0.75, 1.55], ("1", "Pre-registered hypotheses (H1–H6) and earlier leads (L1–L4), each as gain over sex."),
                    "H6 uses n = 25 after OpenSim QA; H1–H5 are one Holm family, H6 its own. L = leads from earlier analyses, judged by the same rule."),
             figure(f2, 2, "Pre-registered variables vs outcome. Each pre-registered variable (H1–H6) and earlier lead (L1, L4) against its outcome, with ordinary least-squares fits within each sex (grey men, orange women). Panel titles give the gain in leave-one-out R² over sex alone and its within-sex permutation p."),
             para("4.4 Within runners", H2),
             para(f"Across {n['n_steps']:,} steps, step speed rises {abs(n['mm_tr']):.3f} m/s for each degree the trunk straightens "
                  f"(95 % CI {mm.loc['trunk_deg_w', 'ci_lo']:+.3f} to {mm.loc['trunk_deg_w', 'ci_hi']:+.3f}) and {n['mm_f']:.2f} m/s per Hz of "
                  f"step frequency within a runner. Runners differ little in that slope (SD {mm.loc['slope SD (trunk_w)', 'coef']:.3f}); "
                  f"those who gain more speed per degree tend to be faster for their sex "
                  f"(r = {mm.loc['r(runner slope, v_peak_ms | sex)', 'coef']:+.2f}, p = {mm.loc['r(runner slope, v_peak_ms | sex)', 'p']:.3f}, exploratory). "
                  f"Muscle–tendon stretch over the top-speed stride is largest in the gluteals and hamstrings (Figure 3); no share predicts "
                  f"speed after correction."),
             figure(f3, 3, f"Within-runner step speed and muscle–tendon excursion. (A) Step speed against trunk lean, both centred on each runner's mean ({n['n_steps']:,} steps; mixed-model slope {n['mm_tr']:+.3f} m/s per degree). (B) Share of whole-body muscle–tendon excursion by muscle group over the top-speed stride (gait2392 trunk and legs; kinematic, not activation; n = {len(sh)})."),
             para("5. Evidence check", H1),
             para("Each claim is set against the sources of the manuscript draft (abstracts retrieved by DOI); † marks a method or "
                  "instrument reference outside that list. The assistant's own earlier claims are included (E12, E17)."),
             *table([["", "Claim", "Our evidence", "Source", "What the source says", "Verdict"]] +
                    [[r_.id, r_.claim, r_.our_evidence, r_.sources, r_.source_says, r_.verdict] for r_ in ev.itertuples()],
                    [0.38, 1.12, 1.1, 0.9, 1.85, 1.15], ("2", "Evidence check.")),
             para("6. Discussion and limits", H1),
             para("What the data establish: sex dominates between-runner top speed in this cohort; published in-sample fits shrink to that "
                  "baseline under honest validation; the pipeline measures angles and speed as Xsens does; and within each run the trunk "
                  "rises and steps quicken as speed builds, in line with step-to-step phase analyses. What they only suggest: forward trunk "
                  "lean out of the blocks, leg-length-normalised step frequency, shank angle at touchdown and arm–leg coordination, each "
                  "hypothesis-generating and each partly carried by single runners. What they cannot address: force orientation, joint "
                  "kinetics and muscle activation (no force plates, body mass or EMG), and contact and touchdown timing finer than one frame "
                  "(16.7 ms at 60 Hz). With 30 runners the smallest between-runner correlation detectable at 80 % power is about r = 0.50; "
                  "the nulls here bound the effects rather than rule them out. The draft's methods should name Ridge, Lasso and Elastic Net "
                  "as penalised linear regressions, not support vector regression, nest any model selection inside its cross-validation, "
                  "report the analysed sample (31 recorded, 30 analysed) rather than the planned 40, and cite Vellucci &amp; Beaudette as "
                  "2023 (published 3 February 2023, volume 4)."),
             para("References", H1)]
    refs = ["Čoh, M., Hébert-Losier, K., Štuhec, S., Babić, V., &amp; Supej, M. (2018). Kinematics of Usain Bolt's maximal sprint velocity. Kinesiology, 50(2), 172–180.",
            "Colyer, S. L., Nagahara, R., Takai, Y., &amp; Salo, A. I. T. (2018). How sprinters accelerate beyond the velocity plateau of soccer players. Scand J Med Sci Sports, 28(12), 2527–2535.",
            "Cotton, R. J., &amp; Leonard, T. (2026). BiomechAgent: AI-assisted biomechanical analysis through code-generating agents. arXiv:2602.06975. †",
            "Debaere, S., Delecluse, C., Aerenhouts, D., Hagman, F., &amp; Jonkers, I. (2013). From block clearance to sprint running. J Sports Sci, 31(2), 137–149.",
            "Debaere, S., Delecluse, C., Aerenhouts, D., Hagman, F., &amp; Jonkers, I. (2015). Control of propulsion and body lift during the first two stances of sprint running. J Sports Sci, 33(19), 2016–2024.",
            "Delp, S. L., et al. (2007). OpenSim: open-source software to create and analyze dynamic simulations of movement. IEEE Trans Biomed Eng, 54(11), 1940–1950. †",
            "Higashihara, A., Nagano, Y., Ono, T., &amp; Fukubayashi, T. (2018). Differences in hamstring activation characteristics between the acceleration and maximum-speed phases of sprinting. J Sports Sci, 36(12), 1313–1318.",
            "Hof, A. L. (1996). Scaling gait data to body size. Gait &amp; Posture, 4(3), 222–223. †",
            "Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scand J Stat, 6(2), 65–70. †",
            "Morin, J.-B., Edouard, P., &amp; Samozino, P. (2011). Technical ability of force application as a determinant factor of sprint performance. Med Sci Sports Exerc, 43(9), 1680–1688.",
            "Morin, J.-B., Bourdin, M., Edouard, P., Peyrot, N., Samozino, P., &amp; Lacour, J.-R. (2012). Mechanical determinants of 100-m sprint running performance. Eur J Appl Physiol, 112(11), 3921–3930.",
            "Rabita, G., et al. (2015). Sprint mechanics in world-class athletes. Scand J Med Sci Sports, 25(5), 583–594.",
            "Schache, A. G., et al. (2011). Effect of running speed on lower limb joint kinetics. Med Sci Sports Exerc, 43(7), 1260–1271.",
            "Schepers, M., Giuberti, M., &amp; Bellusci, G. (2018). Xsens MVN: consistent tracking of human motion using inertial sensing. Xsens Technologies. †",
            "Vellucci, C. L., &amp; Beaudette, S. M. (2023). A need for speed: objectively identifying full-body kinematic and neuromuscular features associated with faster sprint velocities. Front Sports Act Living, 4, 1094163.",
            "von Lieres und Wilkau, H. C., Irwin, G., Bezodis, N. E., Simpson, S., &amp; Bezodis, I. N. (2020). Phase analysis in maximal sprinting. Sports Biomech, 19(2), 141–156.",
            "Weyand, P. G., Sandell, R. F., Prime, D. N. L., &amp; Bundle, M. W. (2010). The biological limits to running speed are imposed from the ground up. J Appl Physiol, 108(4), 950–961."]
    story += [para(r_, REF) for r_ in refs]
    story += [para("Appendix. Reproduce", H1),
              para("python sprint_report.py (workbook, about 3 min) · python sprint_opensim.py (muscle–tendon lengths, about 3 min) · "
                   "python sprint_outputs.py --report (this PDF). Every table above is a sheet of Results/sprint_results.xlsx; the "
                   "pre-registration and its three amendments are in README.md.", P_)]

    def footer(cv, doc):
        cv.saveState(); cv.setFont("Sans", 7.5); cv.setFillColor(MUTED)
        cv.drawString(inch, 0.55 * inch, "60 m sprint: what 30 runners can show"); cv.drawRightString(7.5 * inch, 0.55 * inch, str(doc.page))
        cv.restoreState()
    SimpleDocTemplate(str(REPORT), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                      title="Sprint Technique and Speed", author="60 m project").build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {REPORT.name} and the `evidence` sheet")
    return REPORT


if __name__ == "__main__":
    # `--no-trials`: figures + group animations only; `--every 2`: coarser per-athlete videos;
    # `--report`: Sprint_60m_Report.pdf only.
    argv = sys.argv[1:]
    kwargs = {}
    if "--no-trials" in argv:
        kwargs["trials"] = False
    if "--every" in argv:
        kwargs["every"] = int(argv[argv.index("--every") + 1])
    plt.rcParams.update(STYLE)
    if "--report" in argv:                  # the PDF only (reads the workbook; seconds)
        build_report()
    else:
        run_all_outputs(**kwargs)
