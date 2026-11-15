"""sprint_report.py — every reported number, one cohort pass, one workbook.

    python sprint_report.py        # ~2 min: C3D -> Results/sprint_results.xlsx (+ cohort.pkl)

`sprint_analysis.ipynb` reads what this writes (`load()`), so the notebook opens in
seconds. The preprocessing and models live in `sprint_pipeline.py`; this file only
runs them for the cohort and adds the statistics that used to sit inline in the
old notebooks 03/04/05/08.

Workbook sheets (runner_id first wherever a row is one runner):
    athletes     one row per runner: sex, stature, v_peak, gait timing, 22 angle scalars, PC1-6
    qa           one row per runner: frames, alignment tilt, strides kept/rejected, phase spread
    screen       one row per variable: pooled / within-sex / partial-on-sex r, FDR q, LOO gain over sex
    fpca         one row per component: variance, dominant angle, r with v_peak
    spm          pre-specified SPM clusters (knee_R, hip_R, trunk_lean)
    models       honest LOO ladder, every row also scored against sex alone
    leakage      reported vs honest score for each common shortcut, with shuffle nulls
    reliability  ICC(1,1) as more strides are pooled
    muscle_mtu   one row per runner: OpenSim IK agreement + muscle-tendon stretch (kinematic only; `python sprint_opensim.py`)
    accel        split-time ladder: t10/t20/t30 (first movement -> 10/20/30 m of pelvis travel) from
                 acceleration-phase kinematics; screen_accel = the variable screen for t20
    contacts     one row per runner: every contact-time definition in the codebase, side by side;
                 contact_defs = outcome-blind quality of each definition (plausibility, reliability)
    xsens        one row per runner: pipeline vs Xsens MVN's own model (CoM speed, ISB joint angles, foot contacts)
    eval         BiomechAgent-style scorecard (Cotton & Leonard 2026, arXiv:2602.06975): task category,
                 tolerance, pass rate against a reference the pipeline never reads
"""

from __future__ import annotations

import itertools
import pickle

import numpy as np
import pandas as pd
import scipy.stats as st
import time
from scipy.signal import find_peaks, hilbert
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression, RidgeCV

import sprint_pipeline as SP
from project_paths import PATHS

N_PERM = 5_000
SEED = 1
SPM_ANGLES = ("knee_R", "hip_R", "trunk_lean")    # pre-specified; never widen post hoc
REF = ("SB25", 531, 8.506)                         # regression guard: runner, frames, raw peak m/s
XLSX = PATHS.results / "sprint_results.xlsx"
CACHE = PATHS.results / "cohort.pkl"
XSENS_DIR = SP.C3D_DIR.parent / "Sprint Trials in xlsx"   # Xsens MVN 2025 exports; never used upstream
# Expected ranges (BiomechAgent's "domain instructions"): adult sprinters at maximal speed.
# ponytail: literature-plausible bands, not norms; tighten if a reference cohort is added.
RANGES = {"height_m": (1.45, 2.10), "v_peak_ms": (6.0, 12.5), "t_contact_s": (0.07, 0.16),
          "t_flight_s": (0.08, 0.22), "f_step_hz": (3.5, 5.5), "knee_hi_ROM": (60, 150),
          "t10_s": (1.6, 2.6), "t20_s": (2.7, 4.2), "t30_s": (3.8, 5.6),
          # pre-registered variables: plausibility bands for maximal sprinting (foot must land AHEAD of the pelvis)
          "td_dist_m": (0.10, 0.60), "shank_td_deg": (0, 70), "thigh_retract_dps": (100, 1500),
          "knee_ext_dps": (300, 2000), "crp_armleg_deg": (0, 90), "bflh_lengthen_Lps": (0.2, 5.0)}
SPLITS = (10, 20, 30)          # m of pelvis travel from first movement (frame 0 = hands leave the track)
CONTACT_BAND_M = SP._CONTACT_BAND_M   # both marker definitions: foot "down" within 3 cm of its ground level
# Gait timing scales with body size (Hof 1996 dynamic similarity, confirmed in the `size` sheet), so it is screened
# dimensionless: f·sqrt(L/g), t·sqrt(g/L), L = leg length (taped hip height). Angles are size-free already.
# Performance targets stay ABSOLUTE: within sex they do not scale with size here, so normalizing them would add size.
GAIT = ["height_m", "leg_m", "t_contact_hat", "t_flight_hat", "f_step_hat", "trunk_accel_deg"]
G = 9.81
FOOT_R = [52, 53, 54, 55, 56, 57]   # raw64 right-foot landmarks: heel, MT1, MT5, pivot, heel centre, toe
FOOT_L = [58, 59, 60, 61, 62, 63]
CONTACT_SPEED_MS = 1.5              # contact-point definition: lowest landmark moves < 1.5 m/s (fixed before running)
BALLISTIC_TOL_M = 0.002             # ballistic definition: CoM z fits a -g parabola within 2 mm (fixed before running)
FRONTAL = {"hip_R_abd": "Right Hip Abduction/Adduction", "hip_R_rot": "Right Hip Internal/External Rotation",
           "hip_L_abd": "Left Hip Abduction/Adduction", "hip_L_rot": "Left Hip Internal/External Rotation",
           "L5S1_lat": "L5S1 Lateral Bending", "L5S1_axial": "L5S1 Axial Bending"}
# Pre-registered in README (2026-10-11): id, target, base covariates, added variable, predicted sign (0 = two-sided)
HYPOTHESES = [("H1", "v_peak_ms", ["sex_m"], "thigh_retract_dps", +1), ("H2", "v_peak_ms", ["sex_m"], "knee_ext_dps", +1),
              ("H3", "v_peak_ms", ["sex_m", "leg_m"], "td_dist_m", -1), ("H4", "t10_s", ["sex_m"], "shank_td_deg", -1),
              ("H5", "v_peak_ms", ["sex_m"], "crp_armleg_deg", 0),
              ("H6", "v_peak_ms", ["sex_m"], "bflh_lengthen_Lps", +1)]       # own family (README): Holm over H1-H5 only
MTU_CSV = PATHS.opensim / "Out" / "mtu_summary.csv"             # written by `python sprint_opensim.py`
EXISTING_LEADS = [("L1", "t10_s", ["sex_m"], "trunk_accel_deg", -1), ("L2", "t20_s", ["sex_m"], "trunk_accel_deg", -1),
                  ("L3", "t30_s", ["sex_m"], "trunk_accel_deg", -1), ("L4", "v_peak_ms", ["sex_m"], "f_step_hat", +1)]


def ridge():
    return RidgeCV(alphas=np.logspace(-3, 4, 60))


# ── Cohort pass ──────────────────────────────────────────────────────────────

def _anthro():
    s = pd.read_excel(PATHS.anthropometrics)
    s.index = s.iloc[:, 0].astype(str).str.split("-").str[0].str.strip()
    return pd.DataFrame({"sex": s["Sex"].astype(str).str.strip().str.upper().str[0], "leg_m": s["Hip Height"] / 100})


def _cycle(angles, a, b):
    return np.column_stack([SP.resample_to_cycle(angles[a:b + 1, j]) for j in range(angles.shape[1])])


def _xsens(rid, f, tr, ang, win):
    """Same frames, Xsens MVN's own model: CoM speed, ISB joint angles, foot-speed contacts."""
    x = pd.read_excel(XSENS_DIR / f"{f.stem}.xlsx", sheet_name=["Joint Angles ZXY", "Center of Mass", "Segment Velocity",
                                                                 "Segment Orientation - Euler"])
    s0, n, fs = tr["start"], tr["n_frames"], tr["fs"]
    ja, com, sv, eu = (d.iloc[s0:s0 + n].reset_index(drop=True) for d in x.values())
    cyc = lambda s_: np.mean([SP.resample_to_cycle(np.asarray(s_, float)[a:b + 1]) for a, b in win], 0)
    dcom = np.hypot(com["CoM pos x"] - com["CoM pos x"].iloc[0], com["CoM pos y"] - com["CoM pos y"].iloc[0]).to_numpy()
    info = dict(pd.read_excel(XSENS_DIR / f"{f.stem}.xlsx", sheet_name="General Information", header=None).values)
    out = {"runner_id": rid, "processing": info.get("Processing Quality", "?"),
           "n_frames_c3d": tr["n_raw"], "n_frames_xlsx": len(x["Center of Mass"]),
           **{f"t{m}_xsens_s": _split(dcom, m, fs) for m in SPLITS},
           "v_peak_xsens_ms": SP.peak_velocity_smoothed(np.hypot(com["CoM vel x"], com["CoM vel y"]).to_numpy())[0]}
    for j, col in (("knee_R", "Right Knee Flexion/Extension"), ("hip_R", "Right Hip Flexion/Extension"),
                   ("ankle_R", "Right Ankle Dorsiflexion/Plantarflexion")):
        p, q = cyc(ang[:, SP.ANGLE_NAMES.index(j)]), cyc(ja[col])
        out |= {f"{j}_r": np.corrcoef(p, q)[0, 1], f"{j}_offset_deg": (p - q).mean(),
                f"{j}_ROM_pipe": np.ptp(p), f"{j}_ROM_xsens": np.ptp(q)}
    # right-foot contacts: pipeline = foot-height minima; Xsens = foot horizontal-speed minima
    c_p = find_peaks(-tr["mk"][:, SP.IDX_R_FOOT, SP._VERT], distance=SP.ANGLE_STRIDE_MIN_DISTANCE,
                     prominence=SP.STRIDE_PROMINENCE_M)[0]
    c_x = find_peaks(-np.hypot(sv["Right Foot x"], sv["Right Foot y"]).to_numpy(),
                     distance=SP.ANGLE_STRIDE_MIN_DISTANCE, prominence=1.0)[0]
    span = c_x[(c_x >= win[0][0] - 3) & (c_x <= win[-1][1] + 3)]
    out |= {"n_contacts_R_pipe": len(c_p), "n_contacts_R_xsens": len(c_x),
            "t_stride_pipe_s": np.mean([(b - a) / fs for a, b in win]),
            "t_stride_xsens_s": np.diff(span).mean() / fs if len(span) > 1 else np.nan}
    # exploratory frontal/transverse set (pre-registered as exploratory): cycle mean and ROM; pelvis roll/yaw ROM only
    for k, col in FRONTAL.items():
        q = cyc(ja[col]); out |= {f"fx_{k}_mean": q.mean(), f"fx_{k}_ROM": np.ptp(q)}
    for k, col in (("pelvis_roll", "Pelvis x"), ("pelvis_yaw", "Pelvis z")):
        out[f"fx_{k}_ROM"] = np.ptp(cyc(np.unwrap(np.radians(eu[col])) * 180 / np.pi))
    return out, np.hypot(sv["Right Foot x"], sv["Right Foot y"]).to_numpy(), com["CoM pos z"].to_numpy()


def _crossing_duration(sig, thr, i0, fs):
    """Duration (s) of the run below `thr` that contains frame i0, with both crossings linearly interpolated."""
    if not 0 < i0 < len(sig) - 1 or sig[i0] >= thr:
        return np.nan
    lo, hi = i0, i0
    while lo > 0 and sig[lo - 1] < thr:
        lo -= 1
    while hi < len(sig) - 1 and sig[hi + 1] < thr:
        hi += 1
    if lo == 0 or hi == len(sig) - 1:
        return np.nan
    t0 = lo - 1 + (sig[lo - 1] - thr) / (sig[lo - 1] - sig[lo])
    t1 = hi + (sig[hi] - thr) / (sig[hi] - sig[hi + 1])
    return (t1 - t0) / fs


def _ballistic_flight(z, a, b, fs, tol=BALLISTIC_TOL_M):
    """Longest window in [a, b] where CoM height is a free-fall parabola (z + g·t²/2 linear within `tol`)."""
    t = np.arange(a, b + 1) / fs
    zb = z[a:b + 1] + G * t ** 2 / 2
    best = 0
    for i in range(len(t)):
        for j in range(i + 3 + best, len(t) + 1):           # only windows longer than the current best
            A = np.column_stack([np.ones(j - i), t[i:j]])
            if np.abs(zb[i:j] - A @ np.linalg.lstsq(A, zb[i:j], rcond=None)[0]).max() < tol:
                best = j - i - 1
    return best / fs


def _touchdown(z, i0):
    """Sub-frame touchdown: foot height crosses a LOCAL floor (min within ±10 frames of contact i0) + band.

    Amendment (README): the trial-wide floor drifts 3-5 cm along 60 m, which put touchdown up to 3 frames late
    (0.14 m of pelvis travel per frame at top speed). Returns a fractional frame index, or None.
    """
    lo, hi = max(i0 - 10, 1), min(i0 + 11, len(z))
    thr = z[lo:hi].min() + CONTACT_BAND_M
    i = lo + int(np.argmin(z[lo:hi]))
    while i > lo and z[i - 1] < thr:
        i -= 1
    if i <= lo:
        return None
    return i - 1 + (z[i - 1] - thr) / (z[i - 1] - z[i])


def _at(arr, t):
    """Linear interpolation of arr (frames x ...) at fractional frame t."""
    i = int(np.floor(t)); w = t - i
    return (1 - w) * arr[i] + w * arr[min(i + 1, len(arr) - 1)]


def _split(d, m, fs):
    """Time (s) at which travel distance `d` first reaches m metres, linearly interpolated."""
    i = int(np.argmax(d >= m))
    return (i - 1 + (m - d[i - 1]) / (d[i] - d[i - 1])) / fs


def load_cohort():
    """Read every C3D once. Returns per-runner tables plus the curves the figures need."""
    anthro = _anthro()
    rows, qa, strides, xs, contacts, step_rows, cyc, acc, vel = [], [], [], [], [], [], {}, {}, {}
    vel_raw, raw_top, raw_acc = {}, {}, {}           # for sprint_outputs' figures (drawing only)
    for f in sorted(SP.C3D_DIR.glob("*.c3d")):
        rid = f.stem.split("-")[0].strip()
        if rid in SP.EXCLUDED_PIDS:
            continue
        tr = SP.clean_for_angles(f)
        r64, mk, pk, fs = tr["raw64"], tr["mk"], tr["peak_frame"], tr["fs"]
        if rid == REF[0]:
            assert (tr["n_frames"], round(tr["peak_vel_ms"], 3)) == REF[1:], "SB25 invariant moved"
        cy, _n, spread = SP.stride_cycle_angles(r64, mk, pk)
        if cy is None:
            print(f"skipped {rid}: no valid top-speed stride")
            continue
        cyc[rid] = cy
        win, rejected = SP.find_top_speed_strides(mk, pk)
        v_raw = SP.compute_velocity(mk, fs, raw64=r64)[0]
        v_peak, v_sm = SP.peak_velocity_smoothed(v_raw)
        vel[rid], vel_raw[rid] = (v_sm, fs), v_raw
        t_step = np.mean([(b - a) / fs for a, b in win]) / 2

        # Acceleration: alternating steps 1-2-3 smear phase (old L8); same-foot steps 1-3-5 do not.
        steps, _ = SP.find_first_steps(mk, n_steps=6)
        alt = SP.stride_cycle_angles(r64, mk, pk, windows=steps[:3]) if len(steps) >= 3 else (None, 0, np.nan)
        par = SP.stride_cycle_angles(r64, mk, pk, windows=steps[0:5:2]) if len(steps) >= 5 else (None, 0, np.nan)
        hs = SP.height_scale_for(rid, 1.75)                              # skeleton drawing only
        raw_top[rid] = SP.stride_cycle_raw64(r64, mk, pk, hscale=hs)
        if par[0] is not None:
            acc[rid] = par[0]
            raw_acc[rid] = SP.stride_cycle_raw64(r64, mk, pk, hscale=hs, windows=steps[0:5:2])

        # Every valid top-speed stride, nearest peak first, for the reliability sweep.
        allw, _ = SP.find_top_speed_strides(mk, pk, n_strides=999)
        ang = SP.compute_joint_angles(r64)[0]
        xrow, foot_v, com_z = _xsens(rid, f, tr, ang, win)
        xs.append(xrow)

        # Contact time, every definition in the codebase, per top-speed stride (right foot, same windows):
        #   centroid_band  old 08 / SP.foot_contact_events: Xsens FOOT SEGMENT CENTROID within 3 cm of its trough
        #   heel_toe_band  old 04/05: lower of heel/toe LANDMARK within 3 cm of the trial's ground (2nd pct)
        #   xsens_speed    Xsens foot horizontal speed < 1 m/s (foot ~stationary on the track)
        #   contact_point  lowest right-foot landmark: horizontal speed < CONTACT_SPEED_MS, sub-frame crossings
        #   ballistic      step - flight, flight = CoM z on a -g parabola (BALLISTIC_TOL_M)
        z_ht = np.minimum(r64[:, SP.MK_R_HEEL, SP._VERT], r64[:, SP.MK_R_TOE, SP._VERT])
        ground = np.percentile(z_ht, 2)
        ev_r = SP.foot_contact_events(mk, "R")
        troughs = [e["contact"] for e in SP.foot_contact_events(mk)]            # both feet, ordered
        zf = r64[:, FOOT_R, SP._VERT]
        low_i = np.array(FOOT_R)[zf.argmin(1)]
        vxy = np.gradient(r64[:, :, :2], 1 / fs, axis=0)
        v_low = np.hypot(*vxy[np.arange(len(r64)), low_i].T)
        pel_x = r64[:, SP.MK_PELVIS_ALL, SP._AP].mean(1)
        td = []
        for k, (a, b) in enumerate(win, 1):
            cen = [(e["toeoff"] - e["touchdown"] + 1) / fs for e in ev_r if a <= e["contact"] < b]
            tro = int(np.argmin(zf.min(1)[max(a - 8, 0):a + 8])) + max(a - 8, 0)
            i0 = int(np.argmin(v_low[tro - 3:tro + 4])) + tro - 3
            c0 = min(troughs, key=lambda t_: abs(t_ - a))                       # step = foot-centre trough to next
            nxt = [t_ for t_ in troughs if t_ > c0 + 5]
            bal = (nxt[0] - c0) / fs - _ballistic_flight(com_z, c0, nxt[0], fs) if nxt else np.nan
            on = _touchdown(z_ht, tro)
            if on is not None:                                                  # H3: foot ahead of pelvis at touchdown
                td.append(_at(r64[:, :, SP._AP], on)[low_i[int(round(on))]] - _at(pel_x, on))
            contacts.append({"runner_id": rid, "stride": k, "centroid_band_s": cen[0] if cen else np.nan,
                             "heel_toe_band_s": (z_ht[a:b + 1] < ground + CONTACT_BAND_M).sum() / fs,
                             "contact_point_s": _crossing_duration(v_low, CONTACT_SPEED_MS, i0, fs),
                             "ballistic_s": bal, "xsens_speed_s": (foot_v[a:b + 1] < 1.0).sum() / fs,
                             # sensitivity sweep of the two new definitions (thresholds shown, not chosen)
                             **{f"contact_point_{t_:g}ms": _crossing_duration(v_low, t_, i0, fs) for t_ in (0.5, 1.0)},
                             **{f"ballistic_{int(t_ * 1000)}mm": (nxt[0] - c0) / fs - _ballistic_flight(com_z, c0, nxt[0], fs, t_)
                                if nxt else np.nan for t_ in (0.001, 0.003, 0.005)}})
        t_contact = np.mean([cc["heel_toe_band_s"] for cc in contacts if cc["runner_id"] == rid])   # analysis definition

        # H4: shank angle (ankle -> knee, from vertical, + = knee ahead) at touchdown of acceleration steps 1-3
        shank = []
        for a, _b in steps[:3]:
            side_r = r64[a, FOOT_R, SP._VERT].min() <= r64[a, FOOT_L, SP._VERT].min()
            zfs = np.minimum(r64[:, SP.MK_R_HEEL if side_r else SP.MK_L_HEEL, SP._VERT],
                             r64[:, SP.MK_R_TOE if side_r else SP.MK_L_TOE, SP._VERT])
            on = _touchdown(zfs, a)
            on = float(a) if on is None else on
            knee = _at(r64[:, SP.MK_R_KNEE if side_r else SP.MK_L_KNEE], on).mean(0)
            ankle = _at(r64[:, SP.MK_R_ANKLE if side_r else SP.MK_L_ANKLE], on).mean(0)
            shank.append(np.degrees(np.arctan2(knee[SP._AP] - ankle[SP._AP], knee[SP._VERT] - ankle[SP._VERT])))

        # Step-level rows for the within/between mixed model: every alternating step of the run
        trunk = ang[:, SP.ANGLE_NAMES.index("trunk_lean")]
        for a, b in SP.find_first_steps(mk, n_steps=999)[0]:
            step_rows.append({"runner_id": rid, "speed_ms": (pel_x[b] - pel_x[a]) * fs / (b - a),
                              "trunk_deg": trunk[a:b + 1].mean(), "f_step_hz": fs / (b - a)})
        for rank, (a, b) in enumerate(sorted(allw, key=lambda w: abs((w[0] + w[1]) / 2 - pk)), 1):
            strides.append({"runner_id": rid, "rank": rank,
                            "pct_vpeak": 100 * v_sm[min((a + b) // 2, len(v_sm) - 1)] / v_sm.max(),
                            **SP.scalar_features(_cycle(ang, a, b))})

        d_pelvis = mk[:, SP.IDX_PELVIS, SP._AP] - mk[0, SP.IDX_PELVIS, SP._AP]
        rows.append({"runner_id": rid, "sex": anthro.sex[rid], "leg_m": anthro.leg_m[rid], **{f"t{m}_s": _split(d_pelvis, m, fs) for m in SPLITS},
                     "height_m": SP.PARTICIPANT_ANTHRO[rid]["body_height"], "v_peak_ms": v_peak,
                     "t_contact_s": t_contact, "t_flight_s": t_step - t_contact, "f_step_hz": 1 / t_step,
                     "t_stride_s": 2 * t_step, "td_dist_m": np.mean(td) if td else np.nan,
                     "shank_td_deg": np.mean(shank) if shank else np.nan,
                     "trunk_accel_deg": np.nan if par[0] is None
                     else par[0][:, SP.ANGLE_NAMES.index("trunk_lean")].mean(),
                     **SP.scalar_features(cy),
                     **({} if par[0] is None else {f"acc_{k}": v_ for k, v_ in SP.scalar_features(par[0]).items()})})
        qa.append({"runner_id": rid, "n_frames": tr["n_frames"], "tilt_deg": abs(tr["tilt_deg"]),
                   "n_strides": len(win), "rejected": rejected, "v_peak_raw_ms": tr["peak_vel_ms"],
                   "v_noise_ms": tr["peak_vel_ms"] - v_peak, "spread_top_pct": spread,
                   "spread_accel_alt_pct": alt[2], "spread_accel_par_pct": par[2]})

    A = pd.DataFrame(rows).set_index("runner_id").sort_index()
    smooth = dict(zip(A.index, SP.smooth_angle_curves(np.stack([cyc[r] for r in A.index]), penalty=1.0)))
    fp = SP.run_angle_fpca(smooth, n_components=6)
    for k in range(fp["scores"].shape[1]):
        A[f"PC{k + 1}"] = pd.Series(fp["scores"][:, k], index=fp["pids"])
    N_ = SP.ANGLE_NAMES
    for r_ in A.index:
        cyc_s, dt = smooth[r_], A.t_stride_s[r_] / 100                 # 101 points over one stride
        A.loc[r_, "thigh_retract_dps"] = -np.gradient(cyc_s[:, N_.index("hip_R")], dt)[70:].min()
        A.loc[r_, "knee_ext_dps"] = -np.gradient(cyc_s[:, N_.index("knee_R")], dt)[50:].min()
        ph = lambda j: np.angle(hilbert(cyc_s[:, N_.index(j)] - cyc_s[:, N_.index(j)].mean()))
        crp = lambda j1, j2: np.abs(np.degrees(np.angle(np.exp(1j * (ph(j1) - ph(j2)))))).mean()
        A.loc[r_, "crp_armleg_deg"] = crp("shoulder_L", "hip_R")
        A.loc[r_, "crp_thighs_deg"] = crp("hip_R", "hip_L")
    A["sex_m"] = (A.sex == "M").astype(float)
    X_ = pd.DataFrame(xs).set_index("runner_id")
    A = A.join(X_[[k for k in X_.columns if k.startswith("fx_")]])
    k = np.sqrt(A.leg_m / G)
    A["f_step_hat"], A["t_contact_hat"], A["t_flight_hat"] = A.f_step_hz * k, A.t_contact_s / k, A.t_flight_s / k
    A.insert(A.columns.get_loc("v_peak_ms") + 1, "v_vs_sex_ms",       # speed relative to own sex mean
             A.v_peak_ms - A.groupby("sex").v_peak_ms.transform("mean"))
    Q = pd.DataFrame(qa).set_index("runner_id").sort_index()
    Q["flags"] = np.where(Q.rejected > 0, "rejected strides; ", "") + \
        np.where(Q.spread_top_pct > 10, "top-speed phase spread >10%; ", "") + \
        np.where(Q.n_strides < 5, "<5 strides; ", "") + \
        np.where(Q.spread_accel_par_pct > 50, "accel 1-3-5 phase spread >50%; ", "")
    Q["flag"] = Q.pop("flags").replace("", "ok")
    return {"athletes": A, "qa": Q, "strides": pd.DataFrame(strides), "xsens": pd.DataFrame(xs),
            "contacts": pd.DataFrame(contacts), "steps": pd.DataFrame(step_rows), "cycles": cyc,
            "smooth": smooth, "accel": acc, "vel": vel, "vel_raw": vel_raw, "fpca": fp,
            "raw_top": raw_top, "raw_acc": raw_acc}


# ── Statistics ───────────────────────────────────────────────────────────────

def loo_r2_many(X, Y):
    """Closed-form OLS LOO R² for every column of Y at once (shuffles share one hat matrix)."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    A = np.column_stack([np.ones(len(Y)), np.asarray(X, float).reshape(len(Y), -1)])
    H = A @ np.linalg.pinv(A.T @ A) @ A.T
    e = (Y - H @ Y) / (1 - np.clip(np.diag(H), None, 1 - 1e-12))[:, None]
    return 1 - (e ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def _resid(v, on):
    D = np.column_stack([np.ones(len(v)), on])
    return v - D @ np.linalg.lstsq(D, v, rcond=None)[0]


def _perm_p(null, obs):
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


def icc11(df, col):
    """Shrout-Fleiss ICC(1,1): between-runner share of a feature's variance."""
    g = df.groupby("runner_id")[col]
    k, n = g.size().mean(), g.ngroups
    msb = k * ((g.mean() - df[col].mean()) ** 2).sum() / (n - 1)
    msw = sum(((x - x.mean()) ** 2).sum() for _, x in g) / (len(df) - n)
    return (msb - msw) / (msb + (k - 1) * msw)


def _ols_p(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    dof = len(y) - A.shape[1]
    se = np.sqrt(np.diag(r @ r / dof * np.linalg.pinv(A.T @ A)))
    return 2 * st.t.sf(np.abs(beta / se), dof)[1:]


def stepwise(X, y, p_in=0.05, p_out=0.10):
    """Conventional bidirectional stepwise OLS (Vellucci & Beaudette 2023)."""
    chosen = []
    for _ in range(4 * X.shape[1]):
        changed = False
        rest = [j for j in range(X.shape[1]) if j not in chosen]
        if rest:
            p = [_ols_p(X[:, chosen + [j]], y)[-1] for j in rest]
            if min(p) < p_in:
                chosen.append(rest[int(np.argmin(p))]); changed = True
        if chosen:
            p = _ols_p(X[:, chosen], y)
            if p.max() > p_out:
                chosen.pop(int(np.argmax(p))); changed = True
        if not changed:
            break
    return chosen


def _r2_in(X, y, chosen):
    if not chosen:
        return 0.0
    return 1 - np.sum(_resid(y, X[:, chosen]) ** 2) / np.sum((y - y.mean()) ** 2)


def _pca95(M):
    mu = M.mean(0)
    _u, s, vt = np.linalg.svd(M - mu, full_matrices=False)
    return mu, vt[:int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), 0.95) + 1)]


def _screen(A, y, sex, variables):
    """Pooled, within-sex and partial-on-sex r for each variable, with FDR q and LOO gain over sex."""
    rows = []
    for v in variables:
        x = A[v].to_numpy(float); m = ~np.isnan(x)
        r, p = st.pearsonr(x[m], y[m])
        rp, pp = st.pearsonr(_resid(x[m], sex[m]), _resid(y[m], sex[m]))
        se = 1 / np.sqrt(m.sum() - 4)
        rows.append({"variable": v, "n": int(m.sum()), "r": r, "p": p,
                     "r_M": st.pearsonr(x[m & (sex == 1)], y[m & (sex == 1)])[0],
                     "r_F": st.pearsonr(x[m & (sex == 0)], y[m & (sex == 0)])[0],
                     "r_part_sex": rp, "ci_lo": np.tanh(np.arctanh(rp) - 1.96 * se),
                     "ci_hi": np.tanh(np.arctanh(rp) + 1.96 * se), "p_part": pp,
                     "dR2_loo_vs_sex": SP.loo_r2_fast(np.column_stack([sex[m], x[m]]), y[m])
                     - SP.loo_r2_fast(sex[m], y[m])})
    s = pd.DataFrame(rows)
    s.insert(s.columns.get_loc("p_part") + 1, "q_part", st.false_discovery_control(s.p_part))
    return s.reindex(s.r_part_sex.abs().sort_values(ascending=False).index)


def _ladder(y, specs, sex, target):
    """Honest LOO ladder; OLS rows get a 5,000-shuffle permutation p; every row is also scored against sex alone."""
    rng = np.random.default_rng(SEED)
    Y = np.column_stack([rng.permutation(y) for _ in range(N_PERM)])   # seed 1: the shuffles behind every ledger p-value
    loo_sex = SP.loo_r2_fast(sex, y)
    rows = []
    for label, X, fn, tf in specs:
        m = SP.fit_metrics(y, SP.loo_predict(X, y, fn, tf))
        p = _perm_p(loo_r2_many(X, Y), m["LOO_R2"]) if fn is None and label != "intercept only" else np.nan
        p_gain = np.nan
        if label.startswith("sex + ") and fn is None:     # shuffle the added columns WITHIN sex: tests the gain, not R² > 0
            add = np.asarray(X, float)[:, 1:]
            null = []
            for _ in range(N_PERM):
                idx = np.arange(len(y))
                for g in (0, 1):
                    k = np.where(sex == g)[0]; idx[k] = rng.permutation(k)
                null.append(SP.loo_r2_fast(np.column_stack([sex, add[idx]]), y))
            p_gain = _perm_p(np.array(null), m["LOO_R2"])
        rows.append({"target": target, "model": label, "loo_r2": m["LOO_R2"], "rmse": m["RMSE"], "mae": m["MAE"],
                     "perm_p": p, "dR2_vs_sex": m["LOO_R2"] - loo_sex, "p_gain_vs_sex": p_gain})
    return pd.DataFrame(rows)


def _gain(A, hid, target, base, var, sign, proc, n_boot=2_000):
    """LOO gain of `var` over `base` for `target`, with every robustness check the pre-registration asks for."""
    d = A[[target, *base, var]].dropna()
    y, B, x = d[target].to_numpy(float), d[base].to_numpy(float), d[var].to_numpy(float)
    sex = d["sex_m"].to_numpy(float)

    def loo_err(X):                                   # per-runner closed-form LOO residuals
        D = np.column_stack([np.ones(len(y)), X])
        H = D @ np.linalg.pinv(D.T @ D) @ D.T
        return (y - H @ y) / (1 - np.clip(np.diag(H), None, 1 - 1e-12))
    tss = ((y - y.mean()) ** 2).sum()
    e0, e1 = loo_err(B), loo_err(np.column_stack([B, x]))
    contrib = (e0 ** 2 - e1 ** 2) / tss               # per-runner share of the gain; sums to ΔLOO R²
    dR2 = contrib.sum()
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(N_PERM):                           # shuffle the added variable within sex
        idx = np.arange(len(y))
        for g in (0, 1):
            k = np.where(sex == g)[0]; idx[k] = rng.permutation(k)
        null.append((e0 ** 2 - loo_err(np.column_stack([B, x[idx]])) ** 2).sum() / tss)
    drop1 = []
    for i in range(len(y)):                           # leave each runner out of the whole analysis
        keep = np.arange(len(y)) != i
        yk, Bk, xk = y[keep], B[keep], x[keep]
        drop1.append(SP.loo_r2_fast(np.column_stack([Bk, xk]), yk) - SP.loo_r2_fast(Bk, yk))
    boot = rng.choice(contrib, (n_boot, len(contrib))).sum(1)
    pr = pd.get_dummies(proc.reindex(d.index), drop_first=True).to_numpy(float)
    dR2_proc = SP.loo_r2_fast(np.column_stack([B, pr, x]), y) - SP.loo_r2_fast(np.column_stack([B, pr]), y)
    r_part = st.pearsonr(_resid(x, B), _resid(y, B))[0]
    se = 1 / np.sqrt(len(y) - B.shape[1] - 3)
    lo_r, hi_r = np.tanh(np.arctanh(r_part) - 1.96 * se), np.tanh(np.arctanh(r_part) + 1.96 * se)
    within = {}
    for lab, g in (("M", 1), ("F", 0)):                # within each sex (n = 15): r and LOO gain over the other covariates
        k = sex == g; Bs = B[k][:, [j for j, b in enumerate(base) if b != "sex_m"]]
        Bs = Bs if Bs.size else np.zeros((k.sum(), 1))
        within[f"r_{lab}"] = st.pearsonr(x[k], y[k])[0]
        within[f"dR2_{lab}"] = SP.loo_r2_fast(np.column_stack([Bs, x[k]]), y[k]) - SP.loo_r2_fast(Bs, y[k])
    return {**within, "r_part_lo": lo_r, "r_part_hi": hi_r,
            "max_plausible_r2": max(lo_r ** 2, hi_r ** 2),      # largest share of residual variance the CI allows
            "id": hid, "target": target, "base": "+".join(base), "variable": var, "n": len(y),
            "predicted_sign": {1: "+", -1: "-", 0: "either"}[sign], "r_part": r_part,
            "sign_ok": bool(sign == 0 or np.sign(r_part) == sign), "dR2": dR2, "p_gain": _perm_p(np.array(null), dR2),
            "dR2_min_drop1": min(drop1), "most_influential": d.index[int(np.argmin(drop1))],
            "boot_lo": np.percentile(boot, 2.5), "boot_hi": np.percentile(boot, 97.5), "dR2_with_processing": dR2_proc}


def tables(c):
    A = c["athletes"]
    mtu = pd.read_csv(MTU_CSV).set_index("runner_id") if MTU_CSV.exists() else pd.DataFrame(index=A.index)
    if len(mtu.columns):                    # IK QA (README amendment 3): poor fits give untrustworthy muscle lengths
        mtu["ik_qa"] = np.where((mtu.knee_R_r_osim >= .95) & (mtu.hip_R_r_osim >= .95) & (mtu.ik_rms_mm <= 35), "ok", "fail")
        mcols = [k for k in mtu.columns if k.startswith(("share_", "bifemlh", "bflh", "glut", "iliacus", "rect_fem", "med_gas", "soleus"))]
        mtu.loc[mtu.ik_qa == "fail", mcols] = np.nan
    A = A.join(mtu[[k for k in mtu.columns if k not in A.columns]])
    y, n = A.v_peak_ms.to_numpy(float), len(A)
    sex = (A.sex == "M").to_numpy(float)
    h = A.height_m.to_numpy(float)
    rids = list(A.index)
    scalars = list(SP.scalar_features(c["cycles"][rids[0]]))
    rng = np.random.default_rng(SEED)
    Y = np.column_stack([rng.permutation(y) for _ in range(N_PERM)])   # seed 1: the shuffles behind every ledger p-value

    screen = _screen(A, y, sex, GAIT + scalars + [f"PC{k}" for k in range(1, 7)]
                     + ["crp_thighs_deg"] + [k for k in A.columns if k.startswith("fx_")])
    S5 = c["strides"][c["strides"]["rank"] <= 5]                      # reliability of each angle scalar over 5 strides
    icc_x = {f_: icc11(S5, f_) for f_ in scalars}
    screen.insert(screen.columns.get_loc("r_part_sex") + 1, "icc_x", screen.variable.map(icc_x))
    screen.insert(screen.columns.get_loc("icc_x") + 1, "r_part_corrected",
                  screen.r_part_sex / np.sqrt(screen.icc_x.clip(lower=0.05)))   # disattenuated for x only (one trial: no ICC for v)

    # fpca
    fp = c["fpca"]
    share = (fp["loadings"] ** 2).sum(1)                       # (pc, angle)
    share = share / share.sum(1, keepdims=True) * 100
    fpca = pd.DataFrame({"pc": [f"PC{k + 1}" for k in range(len(fp["explained_var"]))],
                         "var_pct": fp["explained_var"], "cum_pct": np.cumsum(fp["explained_var"]),
                         "top_angle": [SP.ANGLE_NAMES[j] for j in share.argmax(1)],
                         "top_share_pct": share.max(1)})
    fpca = fpca.merge(screen[["variable", "r", "r_part_sex"]], left_on="pc", right_on="variable").drop(columns="variable")
    fpca.insert(0, "variant", "standardised")
    fu = SP.run_angle_fpca(c["smooth"], n_components=6, warn_threshold=101, standardize=False)   # ledger: untried
    su = (fu["loadings"] ** 2).sum(1); su = su / su.sum(1, keepdims=True) * 100
    sc = pd.DataFrame(fu["scores"], index=fu["pids"]).loc[A.index].to_numpy()
    fpca = pd.concat([fpca, pd.DataFrame({
        "variant": "unstandardised", "pc": [f"PC{k + 1}" for k in range(3)], "var_pct": fu["explained_var"][:3],
        "cum_pct": np.cumsum(fu["explained_var"])[:3], "top_angle": [SP.ANGLE_NAMES[j] for j in su.argmax(1)[:3]],
        "top_share_pct": su.max(1)[:3], "r": [np.corrcoef(sc[:, k], y)[0, 1] for k in range(3)],
        "r_part_sex": [st.pearsonr(_resid(sc[:, k], sex), _resid(y, sex))[0] for k in range(3)]})], ignore_index=True)

    # spm (pooled v_peak, as in the manuscript)
    spm, detail = SP.spm_report(c["smooth"], dict(zip(rids, y)), list(SPM_ANGLES), n_perm=N_PERM, seed=1,
                                verbose=False)
    spm.insert(0, "target", "v_peak (pooled)")
    spm_sex, _ = SP.spm_report(c["smooth"], dict(zip(rids, _resid(y, sex))), list(SPM_ANGLES), n_perm=N_PERM, seed=1,
                               verbose=False)
    acc_s = dict(zip(rids, SP.smooth_angle_curves(np.stack([c["accel"][r] for r in rids]), penalty=1.0)))
    spm_acc, _ = SP.spm_report(acc_s, dict(zip(rids, _resid(A.t10_s.to_numpy(float), sex))), list(SPM_ANGLES),
                               n_perm=N_PERM, seed=1, verbose=False)
    spm = pd.concat([spm, spm_sex.assign(target="v_peak | sex"), spm_acc.assign(target="t10 | sex, accel 1-3-5 cycle")],
                    ignore_index=True)

    # models: pre-specified, honest LOO
    knee = A[["knee_hi_ROM", "knee_lo_ROM"]].to_numpy(float)
    hilo = {r: SP.recode_limbs(c["cycles"][r]) for r in rids}
    specs = [("intercept only", np.zeros((n, 1)), None, None), ("sex", sex[:, None], None, None),
             ("height", h[:, None], None, None), ("knee ROM (hi+lo)", knee, None, None),
             ("knee ROM + height", np.column_stack([knee, h]), None, None),
             ("sex + knee ROM + height", np.column_stack([sex, knee, h]), None, None),
             ("22 scalars, ridge", A[scalars].to_numpy(float), ridge, None),
             ("fPC1-6 R/L, ridge (fold refit)", None, ridge, SP.fpca_fold_transform(c["cycles"], rids, 6)),
             ("fPC1-6 HI/LO, ridge (fold refit)", None, ridge, SP.fpca_fold_transform(hilo, rids, 6)),
             ("fPC1-6 R/L unstandardised, ridge", None, ridge,
              SP.fpca_fold_transform(c["cycles"], rids, 6, standardize=False))]
    models = _ladder(y, specs, sex, "v_peak_ms").drop(columns="target")

    # acceleration: split times from acceleration-phase kinematics (same-foot steps 1-3-5).
    # Pre-specified: forward trunk lean out of the blocks. Shorter split = faster, so r < 0 is "better".
    acc_cols = [k for k in A.columns if k.startswith("acc_")]
    acc_curves = {r: c["accel"][r] for r in rids}
    trunk = A.trunk_accel_deg.to_numpy(float)
    accel = pd.concat([_ladder(A[f"t{m}_s"].to_numpy(float), [
        ("intercept only", np.zeros((n, 1)), None, None), ("sex", sex[:, None], None, None),
        ("height", h[:, None], None, None), ("trunk lean, accel", trunk[:, None], None, None),
        ("sex + trunk lean, accel", np.column_stack([sex, trunk]), None, None),
        ("22 accel scalars, ridge", A[acc_cols].to_numpy(float), ridge, None),
        ("accel fPC1-6, ridge (fold refit)", None, ridge, SP.fpca_fold_transform(acc_curves, rids, 6))],
        sex, f"t{m}_s") for m in SPLITS], ignore_index=True)
    screen_accel = _screen(A, A.t20_s.to_numpy(float), sex, [g for g in GAIT if g != "trunk_accel_deg"] + ["v_peak_ms"] + acc_cols)

    # leakage: what each shortcut reports vs the same choice made inside the fold
    X22 = A[scalars].to_numpy(float)
    leak = []

    def topk(Xs, ys, k):
        return np.argsort(-np.abs([np.corrcoef(Xs[:, j], ys)[0, 1] for j in range(Xs.shape[1])]))[:k]
    sel = topk(X22, y, 3)
    honest = SP.fit_metrics(y, SP.loo_predict(None, y, None, lambda tr, te: (
        X22[np.ix_(tr, topk(X22[tr], y[tr], 3))], X22[np.ix_(te, topk(X22[tr], y[tr], 3))])))["LOO_R2"]
    leak.append(("top-3 |r| selection on all runners", SP.loo_r2_fast(X22[:, sel], y), honest, np.nan, np.nan))

    combos = list(itertools.combinations(range(X22.shape[1]), 3))
    design = lambda cb, rows_: np.column_stack([X22[np.ix_(rows_, list(cb))], h[rows_]])
    best = np.full(N_PERM + 1, -np.inf)
    YY = np.column_stack([y, Y])
    for cb in combos:
        best = np.maximum(best, loo_r2_many(design(cb, np.arange(n)), YY))
    pred = np.zeros(n)
    for i in range(n):
        tr = np.delete(np.arange(n), i)
        win = max(combos, key=lambda cb: SP.loo_r2_fast(design(cb, tr), y[tr]))
        D = np.column_stack([np.ones(n), design(win, np.arange(n))])
        pred[i] = D[i] @ np.linalg.lstsq(D[tr], y[tr], rcond=None)[0]
    leak.append((f"best of {len(combos)} 3-feature+height models", best[0],
                 SP.fit_metrics(y, pred)["LOO_R2"], np.percentile(best[1:], 95), _perm_p(best[1:], best[0])))

    for name, curves in (("R/L", c["cycles"]), ("HI/LO", hilo)):
        rep = SP.fit_metrics(y, SP.loo_predict(SP.fpca_scores_leaky(curves, rids, 6), y, ridge))["LOO_R2"]
        hon = SP.fit_metrics(y, SP.loo_predict(None, y, ridge, SP.fpca_fold_transform(curves, rids, 6)))["LOO_R2"]
        leak.append((f"fPCA basis fit on all runners ({name})", rep, hon, np.nan, np.nan))

    M = np.stack([c["cycles"][r] for r in rids]).reshape(n, -1)
    mu, V = _pca95(M)
    S = (M - mu) @ V.T
    for label, F in (("V&B 2023 pipeline: PCA95 + sex, stepwise", np.column_stack([S, sex])),
                     ("V&B 2023 pipeline: PCA95 only, stepwise", S)):
        use_sex = F.shape[1] > S.shape[1]
        rep = _r2_in(F, y, stepwise(F, y))
        pred = np.zeros(n)
        for i in range(n):
            tr = np.delete(np.arange(n), i)
            mu_t, V_t = _pca95(M[tr])
            Ft, Fe = (M[tr] - mu_t) @ V_t.T, (M[i] - mu_t) @ V_t.T
            if use_sex:
                Ft, Fe = np.column_stack([Ft, sex[tr]]), np.r_[Fe, sex[i]]
            ch = stepwise(Ft, y[tr])
            D = np.column_stack([np.ones(len(tr)), Ft[:, ch]])
            pred[i] = np.r_[1.0, Fe[ch]] @ np.linalg.lstsq(D, y[tr], rcond=None)[0]
        null = np.array([_r2_in(F, Y[:, k], stepwise(F, Y[:, k])) for k in range(N_PERM)])
        leak.append((label + f" ({S.shape[1]} PCs)", rep, SP.fit_metrics(y, pred)["LOO_R2"],
                     np.percentile(null, 95), _perm_p(null, rep)))
    leakage = pd.DataFrame(leak, columns=["shortcut", "reported", "honest", "null_95", "p_null"])
    leakage.insert(3, "inflation", leakage.reported - leakage.honest)

    # reliability: ICC as the stride window reaches back into acceleration
    S_ = c["strides"]
    # size: allometric exponent b in log(y) = a + b·log(leg) + sex. b ≈ 0 -> size-free (keep absolute);
    # b ≈ Hof's ±0.5 -> scales with size (use the dimensionless form or adjust for leg length)
    size = []
    for col, hof in (("v_peak_ms", .5), ("t10_s", .5), ("t20_s", .5), ("t30_s", .5),
                     ("f_step_hz", -.5), ("t_contact_s", .5), ("t_flight_s", .5)):
        D = np.column_stack([np.ones(n), np.log(A.leg_m), sex])
        yl = np.log(A[col].to_numpy(float))
        beta, *_ = np.linalg.lstsq(D, yl, rcond=None)
        r_ = yl - D @ beta
        se = np.sqrt((r_ @ r_ / (n - 3)) * np.linalg.inv(D.T @ D)[1, 1])
        lo, hi = beta[1] - 1.96 * se, beta[1] + 1.96 * se
        size.append({"variable": col, "b_leg": beta[1], "ci_lo": lo, "ci_hi": hi, "hof_b": hof,
                     "verdict": "size-free: keep absolute" if lo <= 0 <= hi and not lo <= hof <= hi
                     else "scales with size: use dimensionless" if lo <= hof <= hi and not lo <= 0 <= hi
                     else "inconclusive at n = 30"})

    # contact-time definitions: outcome-blind quality (never chosen by correlation with speed)
    C = c["contacts"]
    main = ["heel_toe_band_s", "centroid_band_s", "contact_point_s", "ballistic_s"]   # heel/toe = analysis definition
    sweep = [k for k in C.columns if k.startswith(("contact_point_", "ballistic_")) and k not in main] + ["xsens_speed_s"]
    defs = main + sweep
    per = C.groupby("runner_id")[defs].mean()
    contact_defs = pd.DataFrame([{
        "definition": d_, "mean_ms": 1000 * per[d_].mean(), "sd_ms": 1000 * per[d_].std(),
        "min_ms": 1000 * per[d_].min(), "max_ms": 1000 * per[d_].max(), "frames_per_contact": 60 * per[d_].mean(),
        "pct_runners_80_140ms": 100 * per[d_].between(.08, .14).mean(),
        "icc_stride": icc11(C.dropna(subset=[d_]), d_),
        "role": "ANALYSIS" if d_ == main[0] else "sensitivity" if d_ in main else "threshold sweep",
        **{f"r_{o[:-2]}": per[d_].corr(per[o]) for o in main if o != d_}} for d_ in defs])

    rel = []
    for k in (5, 8, 10, 999):
        t = S_[S_["rank"] <= k]
        icc = [icc11(t, f) for f in scalars]
        rel.append({"strides": "all" if k == 999 else k, "n_strides": len(t), "median_icc": np.median(icc),
                    "n_icc_gt_075": int(np.sum(np.array(icc) > 0.75)), "n_icc_lt_050": int(np.sum(np.array(icc) < 0.5)),
                    "mean_pct_vpeak": t.pct_vpeak.mean()})

    # pre-registered hypotheses and existing leads: same test, Holm across H1-H5
    proc = c["xsens"].set_index("runner_id").processing.reindex(A.index)
    pre = pd.DataFrame([_gain(A, *h, proc) for h in HYPOTHESES + EXISTING_LEADS if h[3] in A.columns])
    hmask = pre.id.isin(["H1", "H2", "H3", "H4", "H5"])
    pre.loc[hmask, "p_holm"] = np.minimum(1, np.maximum.accumulate(
        np.sort(pre.loc[hmask, "p_gain"].to_numpy()) * (hmask.sum() - np.arange(hmask.sum())))
    )[np.argsort(np.argsort(pre.loc[hmask, "p_gain"].to_numpy()))]
    pre.loc[pre.id == "H6", "p_holm"] = pre.loc[pre.id == "H6", "p_gain"]   # family of one
    pre["robust"] = (pre.dR2_min_drop1 >= .03) & (pre.boot_lo > 0)
    pre["n_implausible"] = [int((~A[v_].dropna().between(*RANGES[v_])).sum()) if v_ in RANGES else 0 for v_ in pre.variable]
    pre["verdict"] = np.where(pre.id.str.startswith("H"),
        np.where((pre.p_holm < .05) & pre.sign_ok & (pre.dR2 > 0) & pre.robust, "SUPPORTED",
                 np.where((pre.p_gain < .05) & pre.sign_ok & (pre.dR2 > 0), "hypothesis-generating, not robust",
                          "not supported")),
        np.where(pre.robust & (pre.p_gain < .05), "robust lead", "hypothesis-generating, not robust"))
    pre.loc[pre.n_implausible > .1 * len(A), "verdict"] = "not measurable at 60 Hz (QA: implausible values)"
    # contact time: the same test on every definition (heel/toe = analysis; others = sensitivity)
    Ac = A.join(per[main].add_prefix("ct_"))
    contact_rob = pd.DataFrame([_gain(Ac, f"C{i}", "v_peak_ms", ["sex_m"], f"ct_{d_}", -1, proc) for i, d_ in enumerate(main, 1)])

    # step-level mixed model: within-runner (deviation from own mean) vs between-runner (runner mean)
    St = c["steps"].merge(A[["sex_m"]], left_on="runner_id", right_index=True)
    for v_ in ("trunk_deg", "f_step_hz"):
        St[v_ + "_b"] = St.groupby("runner_id")[v_].transform("mean")
        St[v_ + "_w"] = St[v_] - St[v_ + "_b"]
    mm = smf.mixedlm("speed_ms ~ trunk_deg_w + trunk_deg_b + f_step_hz_w + f_step_hz_b + sex_m", St,
                     groups=St["runner_id"]).fit(reml=True)
    ci = mm.conf_int()
    steps_mixed = pd.DataFrame({"term": mm.params.index, "coef": mm.params.values, "ci_lo": ci[0].values,
                                "ci_hi": ci[1].values, "p": mm.pvalues.values}).query("term != 'Group Var'")
    steps_mixed["n_steps"], steps_mixed["n_runners"] = len(St), St.runner_id.nunique()
    steps_mixed.insert(0, "model", "random intercept")
    # random slope: does each runner straighten up at a different rate as they speed up?
    ms = smf.mixedlm("speed_ms ~ trunk_deg_w + trunk_deg_b + f_step_hz_w + f_step_hz_b + sex_m", St,
                     groups=St["runner_id"], re_formula="~trunk_deg_w").fit(reml=True)
    slope = pd.Series({g: ms.params["trunk_deg_w"] + re_.get("trunk_deg_w", 0.0)
                       for g, re_ in ms.random_effects.items()}).reindex(A.index)
    rs = [{"model": "random slope", "term": f"slope SD (trunk_w)",
           "coef": float(np.sqrt(ms.cov_re.loc["trunk_deg_w", "trunk_deg_w"])), "ci_lo": np.nan, "ci_hi": np.nan,
           "p": np.nan}]
    for tgt in ("v_peak_ms", "t10_s"):                 # exploratory: do slopes differ with performance? (sex held)
        r_, p_ = st.pearsonr(_resid(slope.to_numpy(), sex), _resid(A[tgt].to_numpy(float), sex))
        rs.append({"model": "random slope", "term": f"r(runner slope, {tgt} | sex)", "coef": r_,
                   "ci_lo": np.nan, "ci_hi": np.nan, "p": p_})
    steps_mixed = pd.concat([steps_mixed, pd.DataFrame(rs).assign(n_steps=len(St), n_runners=St.runner_id.nunique())],
                            ignore_index=True)
    A_out = A.assign(trunk_slope=slope)

    screen_mtu = _screen(A_out, y, sex, [k for k in A_out.columns if k.startswith("share_") or k.endswith("_Lps")
                                         or k.endswith("_excursion_pct")]) if MTU_CSV.exists() else pd.DataFrame()
    out = {"athletes": A_out.reset_index(), "qa": c["qa"].reset_index(), "screen": screen, "fpca": fpca,
           "spm": spm, "models": models, "leakage": leakage, "reliability": pd.DataFrame(rel),
           "accel": accel, "screen_accel": screen_accel, "contacts": per.reset_index(), "contact_defs": contact_defs,
           "size": pd.DataFrame(size), "prespecified": pre, "contact_robustness": contact_rob,
           "steps_mixed": steps_mixed, "muscle_mtu": mtu.reset_index(), "screen_mtu": screen_mtu}
    out["_spm_detail"] = {a: {k: d[k] for k in ("r", "t_crit", "clusters")} for a, d in detail.items()}
    return out


# ── Build / load ─────────────────────────────────────────────────────────────

def evaluate(c, runtime_s):
    """BiomechAgent's benchmark design, applied to this pipeline instead of to an LLM agent.

    Paper categories kept where this data has a reference the pipeline never reads; tolerances are the
    paper's (±10 % spatiotemporal, ±1 event count, exact match for database facts). The reference here is
    Xsens MVN's model of the SAME sensors, so agreement tests processing, not instrument validity.
    """
    A, X, Q = c["athletes"], c["xsens"].set_index("runner_id"), c["qa"]
    X = X.loc[A.index]
    v, vx = A.v_peak_ms, X.v_peak_xsens_ms
    rows = []

    def add(cat, task, value, tol, ok, detail=""):
        ok = np.atleast_1d(ok)
        rows.append({"category": cat, "task": task, "value": value, "tolerance": tol, "n": len(ok),
                     "pass_rate": float(np.mean(ok)) if ok.dtype == bool else np.nan, "detail": detail})

    c3d = sorted(f.stem for f in SP.C3D_DIR.glob("*.c3d"))
    xl = sorted(f.stem for f in XSENS_DIR.glob("*.xlsx"))
    add("database", "C3D trials on disk", len(c3d), "= 31", len(c3d) == 31)
    add("database", "Xsens xlsx export for every C3D", len(set(c3d) & set(xl)), "= 31", set(c3d) <= set(xl))
    add("database", "runners analysed (SB17 excluded)", len(A), "= 30", len(A) == 30)
    add("database", "C3D frames = xlsx frames", int((X.n_frames_c3d == X.n_frames_xlsx).sum()), "exact",
        (X.n_frames_c3d == X.n_frames_xlsx).to_numpy())
    add("database", "SB25 invariant", f"{Q.loc['SB25', 'n_frames']} fr, {Q.loc['SB25', 'v_peak_raw_ms']:.3f} m/s",
        "531 fr, 8.506 m/s", (Q.loc["SB25", "n_frames"], round(Q.loc["SB25", "v_peak_raw_ms"], 3)) == (531, 8.506))

    d = v - vx
    add("spatiotemporal", "v_peak vs Xsens CoM peak", round(d.mean(), 3), "±10 %", (d.abs() / vx <= .10).to_numpy(),
        f"bias {d.mean():+.2f} m/s, LoA {d.mean() - 1.96 * d.std():+.2f} to {d.mean() + 1.96 * d.std():+.2f}; "
        f"r = {np.corrcoef(v, vx)[0, 1]:.3f}")
    ds = X.t_stride_pipe_s - X.t_stride_xsens_s
    add("spatiotemporal", "top-speed stride time vs Xsens foot-speed contacts", round(ds.mean(), 4), "±10 %",
        (ds.abs() / X.t_stride_xsens_s <= .10).to_numpy(), f"bias {1000 * ds.mean():+.1f} ms")
    for m in SPLITS:
        dt = A[f"t{m}_s"] - X[f"t{m}_xsens_s"]
        add("spatiotemporal", f"{m} m split vs Xsens CoM path", round(dt.mean(), 3), "±10 %",
            (dt.abs() / X[f"t{m}_xsens_s"] <= .10).to_numpy(), f"bias {1000 * dt.mean():+.0f} ms, r = {A[f't{m}_s'].corr(X[f't{m}_xsens_s']):.3f}")
    per = c["contacts"].groupby("runner_id")[["centroid_band_s", "heel_toe_band_s"]].mean().loc[A.index]
    dc = per.centroid_band_s - per.heel_toe_band_s
    add("segmentation", "contact time: centroid vs heel-toe definition", round(1000 * dc.mean(), 1), "±10 %",
        (dc.abs() / per.heel_toe_band_s <= .10).to_numpy(), f"bias {1000 * dc.mean():+.0f} ms; r = {per.corr().iloc[0, 1]:.2f}")
    i, j = np.triu_indices(len(v), 1)
    conc = np.sign(v.to_numpy()[i] - v.to_numpy()[j]) == np.sign(vx.to_numpy()[i] - vx.to_numpy()[j])
    add("classification (2AFC)", "which of two runners is faster: pipeline vs Xsens", round(conc.mean(), 3),
        "agree", conc, f"{len(conc)} pairs")
    lr = SP.loo_predict(A[[k for k in A.columns if k.endswith(("_ROM", "_mean")) and "accel" not in k]],
                        (A.sex == "M").astype(float), lambda: _Clf())
    acc = ((lr > .5) == (A.sex == "M")).to_numpy()
    add("classification (2AFC)", "sex from 22 angle scalars (label hidden, LOO)", round(acc.mean(), 3), "chance 0.5",
        acc, f"binomial p = {st.binomtest(int(acc.sum()), len(acc), .5, alternative='greater').pvalue:.4f}")
    dn = X.n_contacts_R_pipe - X.n_contacts_R_xsens
    add("segmentation", "right-foot contacts over the run vs Xsens", round(dn.abs().mean(), 2), "±1",
        (dn.abs() <= 1).to_numpy(), f"pipeline {X.n_contacts_R_pipe.mean():.1f} vs Xsens {X.n_contacts_R_xsens.mean():.1f}")
    for jn in ("knee_R", "hip_R", "ankle_R"):
        r = X[f"{jn}_r"]
        add("kinematics", f"{jn} cycle vs Xsens ISB angle (r)", round(r.median(), 3), "|r| ≥ 0.95",
            (r.abs() >= .95).to_numpy(), f"offset {X[f'{jn}_offset_deg'].mean():+.1f}° (SD {X[f'{jn}_offset_deg'].std():.1f})")
    dr = X.knee_R_ROM_pipe - X.knee_R_ROM_xsens
    add("kinematics", "knee_R ROM vs Xsens", round(dr.mean(), 2), "±10 %", (dr.abs() / X.knee_R_ROM_xsens <= .10).to_numpy(),
        f"bias {dr.mean():+.1f}°")
    out = pd.DataFrame({k: ~A[k].between(*b) for k, b in RANGES.items() if k in A.columns})
    add("domain ranges", "every runner inside expected ranges", int(out.to_numpy().sum()), "0 out of range",
        ~out.any(axis=1).to_numpy(), "; ".join(f"{r_}: {', '.join(out.columns[out.loc[r_]])}" for r_ in out.index[out.any(axis=1)]))
    add("clinical", "dominant-limb laterality (HI/LO side)", np.nan, "—", np.array([np.nan]),
        "not testable: no reference for limb dominance in this dataset")
    add("visualization", "figures follow conventions (units, consistent colours, labels)", np.nan, "rubric",
        np.array([np.nan]), "scored in sprint_analysis.ipynb §8 (self-judged)")
    add("perception", "values read off figures match tables", np.nan, "—", np.array([np.nan]),
        "needs a blind reader; not scored")
    add("efficiency", "full rebuild wall time (s)", round(runtime_s, 1), "—", np.array([np.nan]), "0 errors")
    return pd.DataFrame(rows)


class _Clf(LogisticRegression):
    """Probability output so `SP.loo_predict` can host a classifier."""

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def build():
    t0 = time.time()
    c = load_cohort()
    assert len(c["athletes"]) == 30, f"expected 30 runners, got {len(c['athletes'])}"
    T = tables(c)
    T["xsens"] = c["xsens"]
    T["eval"] = evaluate(c, time.time() - t0)
    y = c["athletes"].v_peak_ms.to_numpy()
    assert abs(loo_r2_many(c["athletes"].height_m, y)[0] - SP.loo_r2_fast(c["athletes"].height_m, y)) < 1e-9
    PATHS.results.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(XLSX) as w:
        for name, df in T.items():
            if not name.startswith("_"):
                df.round(4).to_excel(w, sheet_name=name, index=False)
    with open(CACHE, "wb") as f:
        pickle.dump((c, T), f)
    print(f"wrote {XLSX.name} ({', '.join(k for k in T if not k.startswith('_'))}) and {CACHE.name}")
    return c, T


def load(rerun=False):
    """(cohort, tables) from the cache; rebuilds when asked or when no cache exists."""
    if rerun or not CACHE.exists():
        return build()
    with open(CACHE, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    build()
