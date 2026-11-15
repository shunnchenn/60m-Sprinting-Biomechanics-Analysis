"""sprint_viewer.py — build sprint_viewer.html, the 60m Sprint UI.

    /opt/anaconda3/bin/python sprint_viewer.py      # writes Pipeline/sprint_viewer.html

One self-contained HTML file: every trial's 22 joint centres (same JOINTS /
bone polylines as the MP4 exporter), quantised to int16 mm about the pelvis at
1.75 m display scale (the draw_at_175 convention — the TRUE pelvis path is
kept, so distances and gaps are honest), plus the smoothed speed curve and the
step events from stride.py. The page scrubs frame by frame with a springy,
momentum-carrying scroll (overscroll bounces back), has side / front / 3D
views (3D orbits and zooms like a C3D viewer), and overlays any second
runner aligned by time or by step. Runners are coloured by top speed on a
blue-to-red heat ramp — IN THIS APP red = faster, the OPPOSITE of the repo's
figure rule, at Shun's request (2026-11-14); paper figures keep blue = faster.

Serve with the `sprint-viewer` entry in .claude/launch.json (port 8780), or
open the file directly.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import percentile_filter

import cohort as C
import sprint_outputs as SO
import sprint_pipeline as SP

OUT = Path(__file__).parent / "sprint_viewer.html"


def _b64(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode()


def runner_blob(run, sex):
    raw, fs = run.trial["raw64"], run.trial["fs"]
    pid = run.runner
    pelv = raw[:, SP.MK_PELVIS_ALL, :].mean(axis=1)          # true pelvis path
    hs = SP.height_scale_for(pid, 1.75)
    # The PCA alignment leaves a residual tilt, so the apparent track level
    # drifts ~0.2 m across the 60 m. Same cure as stride.py: a rolling
    # 10th-percentile floor under the lowest foot point (every stance pins it
    # to the track), smoothed, then used as the local ground reference.
    # Display only; nothing measured is touched.
    feet_z = raw[:, [SP.MK_R_HEEL, SP.MK_L_HEEL, SP.MK_R_TOE, SP.MK_L_TOE], 2].min(axis=1)
    win = int(fs) | 1
    ft = percentile_filter(feet_z, 10, size=win, mode="nearest")
    w2, pad = (int(0.5 * fs) | 1), (int(0.5 * fs) | 1) // 2
    ft = np.convolve(np.pad(ft, pad, mode="edge"), np.ones(w2) / w2, mode="valid")
    shape = (raw - pelv[:, None, :]) * hs                    # all 64 markers, draw_at_175 rescale
    pelv = np.column_stack([pelv[:, :2], (pelv[:, 2] - ft) * hs])   # z above ground
    ground = [c for c in run.contacts if not c.on_blocks]
    tops = [(a.touchdown, b.touchdown) for a, _m, b in C.top_speed_strides(run)]
    h = run.headline
    return {
        "id": pid, "sex": sex, "fs": fs, "F": len(raw),
        "top": h["top_speed_ms"], "steps60": h["steps_to_60m"],
        "stride_m": h["avg_stride_length_m"], "stride_hz": h["avg_stride_frequency_hz"],
        "floor": 0.0,                                        # ground is the reference
        "shape": _b64(np.round(shape * 1000), "<i2"),        # mm, pelvis-relative
        "pelv": _b64(pelv, "<f4"),                           # m, true
        "speed": _b64(np.round(run.speed * 100), "<i2"),     # cm/s
        "steps": [[c.touchdown, c.take_off, 1 if c.foot == "R" else 0] for c in ground],
        "tops": tops,
    }


def build():
    data = C.load()
    blobs = [runner_blob(run, data["features"].loc[rid, "sex"])
             for rid, run in sorted(data["runs"].items())]
    blobs.sort(key=lambda b: -b["top"])
    payload = json.dumps(blobs, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", payload)
    OUT.write_text(html)
    print(f"wrote {OUT.name} ({OUT.stat().st_size/1e6:.2f} MB, {len(blobs)} runners)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>60m Sprint UI</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  :root{
    --plate:#FBFBF8; --plate-line:#E4E5DF; --chrome:#1C2832; --chrome-2:#243442;
    --chrome-3:#16202A; --chrome-line:#33465A; --ink:#202B35; --paper-ink:#5A6670;
    --clay:#B08D57; --chrome-text:#E8EDF2; --chrome-dim:#8FA2B5; --focus:#6FA7E8;
    --ease:cubic-bezier(.2,.7,.3,1);
  }
  *{box-sizing:border-box; margin:0}
  html,body{height:100%}
  body{font-family:'Inter',system-ui,sans-serif; font-weight:400; background:var(--chrome);
       color:var(--chrome-text); display:flex; flex-direction:column; overflow:hidden;
       font-feature-settings:"tnum"; font-variant-numeric:tabular-nums;
       -webkit-font-smoothing:antialiased}
  :focus-visible{outline:2px solid var(--focus); outline-offset:2px}
  /* ── control bar ─────────────────────────────────────────────── */
  header{display:flex; align-items:center; gap:12px; padding:10px 16px; flex-wrap:wrap}
  header h1{font-size:14px; font-weight:600; letter-spacing:-.01em; margin-right:4px}
  header .sub{color:var(--chrome-dim); font-size:12px}
  select{appearance:none; -webkit-appearance:none; background:var(--chrome-2); color:var(--chrome-text);
         border:1px solid var(--chrome-line); border-radius:8px; padding:6px 26px 6px 10px;
         font:inherit; font-size:12.5px; font-weight:500; cursor:pointer;
         background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='9' height='6'%3E%3Cpath d='M1 1l3.5 3.5L8 1' fill='none' stroke='%238FA2B5' stroke-width='1.6' stroke-linecap='round'/%3E%3C/svg%3E");
         background-repeat:no-repeat; background-position:right 9px center;
         transition:border-color .15s var(--ease), background-color .15s var(--ease)}
  select:hover{border-color:#4A617A; background-color:#2A3C4D}
  select:active{transform:scale(.98)}
  .vs{color:var(--chrome-dim); font-size:12px}
  /* Apple-style segmented control: a pill slides to the active option */
  .seg{position:relative; display:flex; background:var(--chrome-3);
       border:1px solid var(--chrome-line); border-radius:8px; padding:2px;
       transition:opacity .2s var(--ease)}
  .seg .pill{position:absolute; top:2px; bottom:2px; left:0; background:#3A5068; border-radius:6px;
             transition:transform .18s var(--ease), width .18s var(--ease)}
  .seg button{position:relative; background:none; border:none; color:var(--chrome-dim);
              font:inherit; font-size:12px; font-weight:500; padding:4px 12px; border-radius:6px;
              cursor:pointer; transition:color .15s var(--ease), transform .1s var(--ease)}
  .seg button:hover{color:var(--chrome-text)}
  .seg button:active{transform:scale(.94)}
  .seg button.on{color:#fff}
  .seg.off{opacity:.3; pointer-events:none}
  .tog{background:var(--chrome-3); border:1px solid var(--chrome-line); border-radius:8px;
       color:var(--chrome-dim); font:inherit; font-size:12px; font-weight:500; padding:5px 12px;
       cursor:pointer; transition:color .15s var(--ease), background-color .15s var(--ease),
       border-color .15s var(--ease), transform .1s var(--ease)}
  .tog:hover{color:var(--chrome-text); border-color:#4A617A}
  .tog:active{transform:scale(.95)}
  .tog.on{background:#3A5068; color:#fff}
  .spacer{flex:1}
  /* ── stage + velocity panel ──────────────────────────────────── */
  #main{flex:1; display:flex; min-height:0}
  #stageWrap{flex:1; position:relative; background:var(--plate); min-width:0}
  #stage{position:absolute; inset:0; width:100%; height:100%; cursor:ew-resize; touch-action:none}
  #stageWrap.threed #stage{cursor:grab}
  #stageWrap.threed #stage:active{cursor:grabbing}
  .hud{position:absolute; top:16px; color:var(--ink); pointer-events:none; line-height:1.2}
  #hudL{left:22px} #hudR{right:22px; text-align:right}
  .hud .who{font-size:12.5px; font-weight:600; margin-bottom:6px}
  .hud .who small{font-weight:400; opacity:.65}
  .hud .stepN{font-size:52px; font-weight:700; letter-spacing:-.035em; line-height:1}
  .hud .stepLbl{font-size:12px; color:var(--paper-ink); margin:2px 0 10px}
  .hud .row{font-size:14.5px; font-weight:600; margin-top:5px}
  .hud .row small{font-size:11.5px; font-weight:400; color:var(--paper-ink); margin-left:5px}
  #hudR .row small{margin:0 5px 0 0}
  .threed .hud .stepLbl, .threed .hud .row small{color:#9FB0BF}
  #gap{position:absolute; bottom:14px; left:22px; font-size:12.5px; color:var(--paper-ink);
       pointer-events:none}
  #gap b{font-weight:600; color:var(--ink)}
  .threed #gap{color:#9FB0BF} .threed #gap b{color:#E8EDF2}
  #note{position:absolute; bottom:14px; right:22px; font-size:11.5px; color:var(--paper-ink);
        pointer-events:none}
  .threed #note{color:#8FA2B5}
  #stageCtl{position:absolute; bottom:40px; right:18px; display:flex; gap:6px}
  #stageCtl button{background:rgba(28,40,50,.82); border:1px solid rgba(143,162,181,.35);
                   color:#E8EDF2; border-radius:8px; min-width:30px; height:30px; padding:0 9px;
                   font:inherit; font-size:14px; cursor:pointer;
                   transition:background-color .15s var(--ease), transform .1s var(--ease)}
  #stageCtl button:hover{background:rgba(46,68,88,.92); transform:translateY(-1px)}
  #stageCtl button:active{transform:translateY(0) scale(.92)}
  #stageCtl #zoomReset{font-size:12px}
  #velWrap{width:290px; position:relative; background:var(--plate);
           border-left:1px solid var(--plate-line); cursor:default; overflow:hidden;
           transition:width .28s var(--ease)}
  #velWrap.noT{transition:none}
  #vel{position:absolute; inset:0; width:100%; height:100%; cursor:ew-resize; touch-action:none}
  #velGrip{position:absolute; left:-2px; top:0; bottom:0; width:8px; cursor:col-resize; z-index:2}
  #velGrip:hover{background:linear-gradient(90deg, transparent 2px, #C8CBC2 2px, #C8CBC2 4px, transparent 4px)}
  /* ── scrubber housing ────────────────────────────────────────── */
  #scrubWrap{position:relative; height:92px; border-top:1px solid var(--chrome-line)}
  #scrubWrap canvas{position:absolute; inset:0; width:100%; height:100%}
  #scrub{cursor:ew-resize; touch-action:none}
  footer{display:flex; align-items:center; gap:14px; padding:7px 16px; font-size:12px;
         color:var(--chrome-dim)}
  footer button{background:var(--chrome-2); border:1px solid var(--chrome-line); color:var(--chrome-text);
                border-radius:8px; width:36px; height:27px; cursor:pointer; font-size:11px;
                display:flex; align-items:center; justify-content:center;
                transition:background-color .15s var(--ease), border-color .15s var(--ease),
                           transform .1s var(--ease)}
  footer button:hover{background:#2E4458; border-color:#4A617A; transform:translateY(-1px)}
  footer button:active{transform:translateY(0) scale(.94)}
  #frameLbl{margin-left:auto}
  kbd{border:1px solid var(--chrome-line); border-radius:4px; padding:0 5px; font:inherit; font-size:10.5px}
  @media (max-width:980px){ #velWrap{display:none} }
  @media (max-width:760px){ .hud .stepN{font-size:38px} header{gap:8px} .sub{display:none} }
  @media (prefers-reduced-motion:reduce){ *{transition:none !important} }
</style>
</head>
<body>
<header>
  <h1>60m Sprint UI</h1><span class="sub">30 athletes, 60 Hz</span>
  <select id="runnerSel"></select>
  <span class="vs">against</span>
  <select id="ghostSel"></select>
  <div class="seg" id="viewSeg"><span class="pill"></span><button data-v="side" class="on">Side</button><button data-v="front">Front</button><button data-v="3d">3D</button></div>
  <div class="seg off" id="modeSeg"><span class="pill"></span><button data-m="track" class="on">Track</button><button data-m="lock">Lock</button></div>
  <div class="seg off" id="alignSeg"><span class="pill"></span><button data-a="time" class="on">Same time</button><button data-a="step">Same step</button></div>
  <button class="tog on" id="velBtn">Velocity</button>
  <div class="spacer"></div><span class="sub" id="runnerInfo"></span>
</header>

<div id="main">
<div id="stageWrap">
  <canvas id="stage"></canvas>
  <div class="hud" id="hudL"></div>
  <div class="hud" id="hudR"></div>
  <div id="gap"></div>
  <div id="note">right side red &middot; left side blue</div>
  <div id="stageCtl">
    <button id="zoomIn" title="zoom in">+</button>
    <button id="zoomOut" title="zoom out">&minus;</button>
    <button id="zoomReset" title="reset the view">Reset</button>
  </div>
</div>
<div id="velWrap"><canvas id="vel"></canvas><div id="velGrip"></div></div>
</div>

<div id="scrubWrap"><canvas id="scrubBase"></canvas><canvas id="scrub"></canvas></div>

<footer>
  <button id="playBtn" title="play / pause (space)">&#9654;</button>
  <select id="rateSel"><option value="0.25">0.25&times;</option><option value="0.5" selected>0.5&times;</option><option value="1">1&times;</option></select>
  <span id="hint">drag the stage or the timeline to scrub, <kbd>&larr;</kbd><kbd>&rarr;</kbd> step one frame, <kbd>space</kbd> plays</span>
  <span id="frameLbl"></span>
</footer>

<script>
const RAW = __DATA__;
// Chris V. 64-marker bone polylines (0-based, from MATLAB xsens_scr_video.m)
const AXIAL=[
  [1,2,4,3,1,5,6,2,6,7,5,7,8,9,10,13,15,20,11,14,10,14,20,14,21,14,11,21,15],
  [21,10],[20,10],[16,17,18,16,19,17,19,18,19,15]];
const RIGHT=[
  [20,22,23,28,23,20],[20,28,26,28,27,26],[23,26,23,27,26],[22,26,22,27,26],
  [26,32,27,32,33,32,34,32,33,26],
  [38,39,38,40,38,41,40,39,41,46,39,46,40,46],
  [46,47,46,48],[39,47,39,48],[40,47,40,48],
  [47,48,52,47,52,53,52,54,52,55,56,55,57,52]];
const LEFT=[
  [21,24,25,24,31,21],[21,25],
  [29,30,24,29,24,30,24,25,29,25,30,25],
  [31,29,30,29,31,30,31],
  [29,35,29,36,29,37,35,36,30,35,30,36,30,37,30],
  [42,43,42,44,42,45,44,43,45],
  [49,45,49,44,49,43,49],[49,50,49,51,49],
  [43,50,43,51,43],[44,50,44,51,44],
  [50,51,50,58,51,58,59,58,59,58,61,58,60,61,62,61,63]];
const NJ=64;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function dec(b64, T){
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) u8[i]=bin.charCodeAt(i);
  return new T(u8.buffer);
}
function smooth(a, w){                       // centred moving average (camera only)
  const n=a.length, half=w>>1, out=new Float32Array(n), cs=new Float64Array(n+1);
  for(let i=0;i<n;i++) cs[i+1]=cs[i]+a[i];
  for(let i=0;i<n;i++){const lo=Math.max(0,i-half), hi=Math.min(n-1,i+half);
    out[i]=(cs[hi+1]-cs[lo])/(hi-lo+1);}
  return out;
}
const R = RAW.map(r => {
  const pelv = dec(r.pelv, Float32Array);
  const camX = smooth(Float32Array.from({length:r.F},(_,i)=>pelv[i*3]), 31);
  return {...r, shape:dec(r.shape,Int16Array), pelv, camX, speed:dec(r.speed,Int16Array)};
});
// Heat ramp over the cohort's top speeds. IN THIS APP red = faster (Shun's
// request); the repo's printed figures keep the opposite rule.
const VMIN = Math.min(...R.map(r=>r.top)), VMAX = Math.max(...R.map(r=>r.top));
const colorOf = (r, dark) =>
  `hsl(${Math.round(220*(1-(r.top-VMIN)/(VMAX-VMIN||1)))},64%,${dark?60:42}%)`;

// ── state ──────────────────────────────────────────────────────
let A = R[0], B = null;                       // primary, comparison
let view = "side", mode = "track", align = "time", playing = false, rate = 0.5;
let f = 0, fd = 0, vF = 0;                    // target frame, displayed frame, fling velocity
let dragScrub = false, needDraw = true;
let yaw = -0.6, pitch = 0.3, dist = 4.6, zoom2D = 1;      // 3D camera + 2D zoom
const OVER = 24;                              // rubber-band overshoot, frames
const clamp = v => Math.max(0, Math.min(A.F-1, v));
const is3D = () => view === "3d";

function applyDelta(d){                       // scrub with resistance past the ends
  let v = f + d;
  const out = v<0 ? -v : v>A.F-1 ? v-(A.F-1) : 0;
  if (out>0){ const k = Math.max(0, 1-out/OVER); v = f + d*k*k; }
  f = Math.min(A.F-1+OVER, Math.max(-OVER, v)); needDraw = true;
}
function seek(v){ f = clamp(v); vF = 0; needDraw = true; }

// joint positions at fractional frame -> [[h,v] x22] in metres (h: AP or ML)
function pose(r, ff, hAxis, trackH){
  const i = Math.min(Math.floor(ff), r.F-2), t = ff - i, J = [];
  const s = r.shape, p = r.pelv;
  const ph = (1-t)*p[i*3+hAxis] + t*p[(i+1)*3+hAxis];
  const pz = (1-t)*p[i*3+2]     + t*p[(i+1)*3+2];
  for (let j=0;j<NJ;j++){
    const a=(i*NJ+j)*3, b=((i+1)*NJ+j)*3;
    const h = ((1-t)*s[a+hAxis] + t*s[b+hAxis])/1000;
    const v = ((1-t)*s[a+2]     + t*s[b+2])/1000 + pz;
    J.push([trackH ? h+ph : h, v]);
  }
  return J;
}
function pose3(r, ff, trackH){               // full 3D: [[ap, ml, z] x22]
  const i = Math.min(Math.floor(ff), r.F-2), t = ff - i, J = [];
  const s = r.shape, p = r.pelv;
  const pap = (1-t)*p[i*3]   + t*p[(i+1)*3];
  const pml = (1-t)*p[i*3+1] + t*p[(i+1)*3+1];
  const pz  = (1-t)*p[i*3+2] + t*p[(i+1)*3+2];
  for (let j=0;j<NJ;j++){
    const a=(i*NJ+j)*3, b=((i+1)*NJ+j)*3;
    const ap = ((1-t)*s[a]   + t*s[b])/1000   + (trackH ? pap : 0);
    const ml = ((1-t)*s[a+1] + t*s[b+1])/1000 + (trackH ? pml : 0);
    const z  = ((1-t)*s[a+2] + t*s[b+2])/1000 + pz;
    J.push([ap, ml, z]);
  }
  return J;
}
const camAt = (r,ff)=>{const i=Math.min(Math.floor(ff),r.F-2),t=ff-i;
  return (1-t)*r.camX[i]+t*r.camX[i+1];};
const pelvAP = (r,ff)=>{const i=Math.min(Math.floor(ff),r.F-2),t=ff-i;
  return (1-t)*r.pelv[i*3]+t*r.pelv[(i+1)*3];};
const speedAt = (r,ff)=>{const i=Math.min(Math.floor(ff),r.F-2),t=ff-i;
  return ((1-t)*r.speed[i]+t*r.speed[i+1])/100;};

// step-progress clock: -1..0 across the block phase, then k..k+1 across step k+1
function progress(r, ff){
  const td = r.steps.map(s=>s[0]);
  if (ff < td[0]) return ff/td[0] - 1;
  let k = td.length-1;
  while (k>0 && td[k]>ff) k--;
  const next = td[k+1] ?? r.F-1;
  return k + Math.min(1,(ff-td[k])/Math.max(1,next-td[k]));
}
function frameAt(r, prog){                    // inverse of progress()
  const td = r.steps.map(s=>s[0]);
  if (prog < 0) return (prog+1)*td[0];
  const k = Math.min(Math.floor(prog), td.length-1);
  const next = td[k+1] ?? r.F-1;
  return Math.min(td[k] + (prog-k)*(next-td[k]), r.F-1);
}
const ghostFrame = ff => align==="time" ? Math.min(ff, B.F-1) : frameAt(B, progress(A, ff));

// ── stage ──────────────────────────────────────────────────────
const stage = document.getElementById("stage"), sctx = stage.getContext("2d");
function fit(cv){ const d=devicePixelRatio||1, w=cv.clientWidth, h=cv.clientHeight;
  if(cv.width!==Math.round(w*d)||cv.height!==Math.round(h*d)){cv.width=Math.round(w*d); cv.height=Math.round(h*d);}
  cv.getContext("2d").setTransform(d,0,0,d,0,0); return [w,h]; }

function strokeBody(ctx, P, color, alpha, lw){   // P: [[x,y] x64] screen px
  // color: one CSS color (ghost), or {ax,r,l} for the primary — the Chris V.
  // mesh with axial ink, right limbs red, left limbs blue.
  const c = typeof color === "string" ? {ax:color, r:color, l:color} : color;
  ctx.lineCap = ctx.lineJoin = "round";
  for (const [polys, a, col] of [[AXIAL,1,c.ax],[RIGHT,1,c.r],[LEFT,1,c.l]]){
    ctx.globalAlpha = alpha*a; ctx.lineWidth = lw; ctx.strokeStyle = col;
    for (const poly of polys){
      ctx.beginPath();
      poly.forEach((j,i)=> i ? ctx.lineTo(P[j][0],P[j][1]) : ctx.moveTo(P[j][0],P[j][1]));
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

function draw2D(ctx, w, h, fc){
  const sc = (h*0.72)/2.1 * zoom2D, baseY = h*0.88;
  const hAxis = view==="side" ? 0 : 1;
  const track = view==="side" && mode==="track";
  const camH = track ? camAt(A, fc) : 0;
  const X = m => w/2 + (m-camH)*sc, Y = m => baseY - m*sc;
  const toPx = J => J.map(([a,b])=>[X(a),Y(b)]);

  // floor + metre ticks (ticks only mean something on the real track)
  ctx.strokeStyle = css("--clay"); ctx.lineWidth = 1.5; ctx.globalAlpha=.85;
  ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(w, Y(0)); ctx.stroke();
  if (track){
    ctx.fillStyle = css("--clay"); ctx.font = "11.5px Inter"; ctx.textAlign="center";
    const m0 = Math.floor((camH - w/2/sc)), m1 = Math.ceil((camH + w/2/sc));
    for (let m=m0; m<=m1; m++){
      const big = m%2===0;
      ctx.globalAlpha = big?.9:.45;
      ctx.fillRect(X(m)-.75, Y(0), 1.5, big?9:5);
      if (big && m>=0) ctx.fillText(m+" m", X(m), Y(0)+24);
    }
  }
  ctx.globalAlpha = 1;

  if (B){
    const gf = ghostFrame(fc);
    strokeBody(ctx, toPx(pose(B, gf, hAxis, track)), colorOf(B), .6, 1.3);
    if (track){                                   // off-screen arrow
      const px = X(pelvAP(B,gf));
      if (px < -20 || px > w+20){
        ctx.fillStyle = colorOf(B); ctx.font="600 13px Inter";
        ctx.textAlign = px<0 ? "left":"right";
        ctx.fillText((px<0?"← ":"")+B.id+(px<0?"":" →"), px<0?14:w-14, h*0.5);
      }
    }
  }
  strokeBody(ctx, toPx(pose(A, fc, hAxis, track)),
             {ax:css("--ink"), r:"#d62728", l:"#1f77b4"}, .9, 1.6);
}

function draw3D(ctx, w, h, fc){
  const g = ctx.createLinearGradient(0,0,0,h);
  g.addColorStop(0,"#232E38"); g.addColorStop(1,"#141C24");
  ctx.fillStyle = g; ctx.fillRect(0,0,w,h);
  const track = mode==="track";
  const tAP = track ? camAt(A, fc) : 0, tML = 0, tZ = 0.95;
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const fl = h*1.05, horizon = h*0.52;
  function proj(ap, ml, z){
    const x = ap-tAP, y = ml-tML, zz = z-tZ;
    const x1 =  x*cy + y*sy;
    const y1 = -x*sy + y*cy;
    const z1 =  zz*cp - y1*sp;
    const d  =  zz*sp + y1*cp;
    const s  = fl / Math.max(dist - d, 0.4);
    return [w/2 + x1*s, horizon - z1*s];
  }
  // C3D-style floor grid, 1 m squares around the camera target
  ctx.lineWidth = 1;
  const a0 = Math.floor(tAP)-5, a1 = Math.floor(tAP)+5;
  for (let a=a0; a<=a1; a++){
    ctx.strokeStyle = a%5===0 ? "#465664" : "#33404C";
    const p=proj(a,-2.5,0), q=proj(a,2.5,0);
    ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke();
  }
  for (let m=-2.5; m<=2.5; m++){
    ctx.strokeStyle = m===0 ? "#465664" : "#33404C";
    const p=proj(a0,m,0), q=proj(a1,m,0);
    ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke();
  }
  const toPx = J => J.map(([a,m,z])=>proj(a,m,z));
  if (B) strokeBody(ctx, toPx(pose3(B, ghostFrame(fc), track)), colorOf(B,true), .6, 1.3);
  strokeBody(ctx, toPx(pose3(A, fc, track)),
             {ax:"#E8EDF2", r:"#E05A5A", l:"#6FA7E8"}, .9, 1.6);
}

function drawStage(){
  const [w,h] = fit(stage), ctx = sctx;
  ctx.clearRect(0,0,w,h);
  const fc = clamp(fd), over = fd - fc;       // overshoot slides the stage, then springs back
  ctx.save(); ctx.translate(-over*2.0, 0);
  if (is3D()) draw3D(ctx, w, h, fc); else draw2D(ctx, w, h, fc);
  ctx.restore();

  // HUD: one metric block per runner — primary left, comparison right
  const dark = is3D();
  function hudHTML(r, ff, right){
    const p = progress(r, ff), inBlocks = p < 0;
    const d = pelvAP(r,ff) - r.pelv[0];
    const row = (v,u) => right ? `<div class="row"><small>${u}</small>${v}</div>`
                               : `<div class="row">${v}<small>${u}</small></div>`;
    return `<div class="who">${r.id} <small>${r.top.toFixed(2)} m/s top, ${r.sex}</small></div>` +
      `<div class="stepN">${inBlocks ? "–" : Math.floor(p)+1}</div>` +
      `<div class="stepLbl">${inBlocks ? "in the blocks" : "steps in"}</div>` +
      row(speedAt(r,ff).toFixed(2), "m/s") + row(d.toFixed(1), "m") + row((ff/r.fs).toFixed(2), "s");
  }
  const hudL = document.getElementById("hudL"), hudR = document.getElementById("hudR");
  hudL.innerHTML = hudHTML(A, fc, false); hudL.style.color = colorOf(A, dark);
  hudR.innerHTML = B ? hudHTML(B, ghostFrame(fc), true) : "";
  if (B) hudR.style.color = colorOf(B, dark);
  const gapEl = document.getElementById("gap");
  if (B && align==="time" && view!=="front"){
    const g = pelvAP(A,fc) - pelvAP(B, Math.min(fc,B.F-1));
    gapEl.innerHTML = `${A.id} is <b>${Math.abs(g).toFixed(1)} m ${g>=0?"ahead of":"behind"}</b> ${B.id}`;
  } else if (B && align==="step"){
    gapEl.innerHTML = `same step compared: each runner shown at its own clock`;
  } else gapEl.textContent = "";
  document.getElementById("frameLbl").textContent = `frame ${Math.round(fc)} / ${A.F-1}`;
  drawVel();
  drawPlayhead();
}

// ── velocity panel: full speed-against-time chart, click to seek ─
const vel = document.getElementById("vel"), velWrap = document.getElementById("velWrap");
let velW = 290, velOpen = true;
function drawVel(){
  if (!velOpen || velWrap.clientWidth < 60) return;
  const [w,h] = fit(vel), ctx = vel.getContext("2d");
  ctx.clearRect(0,0,w,h);
  const L=36, Rm=14, T=30, Bm=32;
  const fc = clamp(fd);
  const tmax = Math.max(A.F, B?B.F:0)/A.fs, vmax = Math.ceil(Math.max(A.top, B?B.top:0))+0.5;
  const X = t => L + t/tmax*(w-L-Rm), Y = v => h-Bm - v/vmax*(h-T-Bm);
  ctx.font = "11px Inter"; ctx.fillStyle = css("--paper-ink");
  ctx.strokeStyle = css("--plate-line"); ctx.lineWidth = 1;
  for (let v=0; v<=vmax; v++){
    ctx.beginPath(); ctx.moveTo(L,Y(v)); ctx.lineTo(w-Rm,Y(v)); ctx.stroke();
    ctx.textAlign="right"; ctx.fillText(v, L-6, Y(v)+3.5);
  }
  for (let t=0; t<=tmax; t+=2){
    if (X(t) > w-Rm-42) break;                      // keep clear of the axis label
    ctx.textAlign="center"; ctx.fillText(t, X(t), h-Bm+16);
  }
  ctx.textAlign="left"; ctx.fillText("speed, m/s", L, T-12);
  ctx.textAlign="right"; ctx.fillText("time, s", w-Rm, h-Bm+16);
  const curve = (r,col,lw)=>{ ctx.beginPath();
    for(let i=0;i<r.F;i++){const x=X(i/r.fs),y=Y(r.speed[i]/100); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
    ctx.strokeStyle=col; ctx.lineWidth=lw; ctx.stroke(); };
  if (B) curve(B, colorOf(B), 1.8);
  curve(A, colorOf(A), 2.2);
  const dot = (r,ff,col)=>{ const x=X(ff/r.fs), y=Y(speedAt(r,ff));
    ctx.fillStyle=col; ctx.beginPath(); ctx.arc(x,y,4,0,7); ctx.fill();
    ctx.font="600 11.5px Inter"; ctx.textAlign="left";
    ctx.fillText(speedAt(r,ff).toFixed(2), x+7, y-6); };
  ctx.strokeStyle="#C9CCC4"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(X(fc/A.fs),T-4); ctx.lineTo(X(fc/A.fs),h-Bm); ctx.stroke();
  if (B) dot(B, ghostFrame(fc), colorOf(B));
  dot(A, fc, colorOf(A));
}
vel.addEventListener("pointerdown", e => {
  vel.setPointerCapture(e.pointerId); stopPlay();
  const go = ev => { const L=36, Rm=14, w=vel.clientWidth;
    const tmax = Math.max(A.F, B?B.F:0)/A.fs;
    seek(((ev.offsetX-L)/(w-L-Rm))*tmax*A.fs); };
  go(e); vel.onpointermove = go;
  vel.onpointerup = () => vel.onpointermove = null;
});
// show / hide the panel
document.getElementById("velBtn").onclick = e => {
  velOpen = !velOpen;
  e.currentTarget.classList.toggle("on", velOpen);
  velWrap.style.width = velOpen ? velW+"px" : "0px";
};
velWrap.addEventListener("transitionend", ()=>{ needDraw = true; });
// drag the panel edge to resize it
document.getElementById("velGrip").addEventListener("pointerdown", e => {
  e.stopPropagation();
  const grip = e.currentTarget; grip.setPointerCapture(e.pointerId);
  const x0 = e.clientX, w0 = velWrap.clientWidth;
  velWrap.classList.add("noT");
  grip.onpointermove = ev => {
    velW = Math.max(200, Math.min(520, w0 + (x0 - ev.clientX)));
    velWrap.style.width = velW+"px"; needDraw = true;
  };
  grip.onpointerup = () => { grip.onpointermove = null; velWrap.classList.remove("noT"); };
});

// ── scrubber ───────────────────────────────────────────────────
const sbase = document.getElementById("scrubBase"), scv = document.getElementById("scrub");
function drawScrubBase(){
  const [w,h] = fit(sbase), ctx = sbase.getContext("2d");
  ctx.clearRect(0,0,w,h);
  const X = fr => fr/(A.F-1)*w;
  const vmaxC = Math.max(...R.map(r=>r.top))*100;
  const Y = cs => h*0.86 - (cs/vmaxC)*h*0.62;
  ctx.fillStyle = css("--chrome-2");
  ctx.fillRect(0,0,X(A.steps[0][0]),h);
  ctx.fillStyle = "#8FB6DC"; ctx.globalAlpha = .16;        // the five top-speed strides
  for (const [a,b] of A.tops) ctx.fillRect(X(a),0,X(b)-X(a),h);
  ctx.globalAlpha = 1;
  ctx.beginPath(); ctx.moveTo(0,h*0.86);
  for (let i=0;i<A.F;i++) ctx.lineTo(X(i), Y(A.speed[i]));
  ctx.lineTo(w,h*0.86); ctx.closePath();
  ctx.fillStyle = "#2E4256"; ctx.fill();
  ctx.beginPath();
  for (let i=0;i<A.F;i++) i ? ctx.lineTo(X(i),Y(A.speed[i])) : ctx.moveTo(X(i),Y(A.speed[i]));
  ctx.strokeStyle = colorOf(A, true); ctx.lineWidth = 1.6; ctx.stroke();
  if (B){
    ctx.beginPath();
    const XB = fr => (align==="time" ? fr/(A.F-1) : frameAt(A, progress(B, fr))/(A.F-1)) * w;
    for (let i=0;i<B.F;i++){const x=XB(i); i?ctx.lineTo(x,Y(B.speed[i])):ctx.moveTo(x,Y(B.speed[i]));}
    ctx.strokeStyle = colorOf(B, true);
    ctx.globalAlpha=.8; ctx.stroke(); ctx.globalAlpha=1;
  }
  ctx.strokeStyle = css("--chrome-dim"); ctx.lineWidth = 1.2;  // step ticks: R up, L down
  for (const [td,,isR] of A.steps){
    ctx.beginPath(); ctx.moveTo(X(td), h*0.86 + (isR?-7:1));
    ctx.lineTo(X(td), h*0.86 + (isR?-1:7)); ctx.stroke();
  }
  ctx.strokeStyle = css("--chrome-line");
  ctx.beginPath(); ctx.moveTo(0,h*0.86); ctx.lineTo(w,h*0.86); ctx.stroke();
}
function drawPlayhead(){
  const [w,h] = fit(scv), ctx = scv.getContext("2d");
  ctx.clearRect(0,0,w,h);
  const x = clamp(fd)/(A.F-1)*w;
  ctx.fillStyle = css("--chrome-text");
  ctx.fillRect(x-.75, 0, 1.5, h);
  ctx.beginPath(); ctx.arc(x, h*0.86, 4, 0, 7); ctx.fill();
}

// ── interaction ────────────────────────────────────────────────
scv.addEventListener("pointerdown", e => { scv.setPointerCapture(e.pointerId);
  stopPlay(); dragScrub = true; vF = 0;
  const go = ev => { f = clamp((ev.offsetX/scv.clientWidth)*(A.F-1)); needDraw = true; };
  go(e); scv.onpointermove = go;
  scv.onpointerup = () => { scv.onpointermove = null; dragScrub = false; }; });

stage.addEventListener("pointerdown", e => { stage.setPointerCapture(e.pointerId);
  let lx = e.clientX, ly = e.clientY, lt = performance.now();
  if (is3D()){
    stage.onpointermove = ev => {
      yaw += (ev.clientX-lx)*0.008; pitch += (ev.clientY-ly)*0.008;
      pitch = Math.max(-0.1, Math.min(1.35, pitch));
      lx = ev.clientX; ly = ev.clientY; needDraw = true;
    };
    stage.onpointerup = () => stage.onpointermove = null;
    return;
  }
  stopPlay(); dragScrub = true; vF = 0;
  stage.onpointermove = ev => {
    const now = performance.now(), dt = Math.max((now-lt)/1000, 1e-3);
    const d = (ev.clientX-lx)*0.22;
    applyDelta(d);
    vF = 0.72*vF + 0.28*(d/dt);               // momentum carried to release
    lx = ev.clientX; lt = now;
  };
  stage.onpointerup = () => { stage.onpointermove = null; dragScrub = false; };
});
stage.addEventListener("wheel", e => { e.preventDefault();
  if (is3D()){ dist = Math.max(1.6, Math.min(14, dist*Math.exp(e.deltaY*0.0012))); needDraw = true; return; }
  stopPlay(); vF = 0; applyDelta(e.deltaY*0.045); }, {passive:false});
addEventListener("keydown", e => {
  if (e.target.tagName==="SELECT") return;
  if (e.code==="Space"){ e.preventDefault(); togglePlay(); }
  if (e.key==="ArrowRight"){ stopPlay(); seek(Math.floor(clamp(fd))+(e.shiftKey?5:1)); }
  if (e.key==="ArrowLeft"){ stopPlay(); seek(Math.ceil(clamp(fd))-(e.shiftKey?5:1)); }
});

// zoom buttons: never get stuck too far in or out
document.getElementById("zoomIn").onclick = () => {
  if (is3D()) dist = Math.max(1.6, dist/1.28); else zoom2D = Math.min(3, zoom2D*1.18);
  needDraw = true; };
document.getElementById("zoomOut").onclick = () => {
  if (is3D()) dist = Math.min(14, dist*1.28); else zoom2D = Math.max(.5, zoom2D/1.18);
  needDraw = true; };
document.getElementById("zoomReset").onclick = () => {
  yaw=-0.6; pitch=0.3; dist=4.6; zoom2D=1; needDraw = true; };

// ── the clock: playback, momentum, and the spring that makes it fluid ─
const playBtn = document.getElementById("playBtn");
let lastT = 0;
function loop(t){
  requestAnimationFrame(loop);
  const dt = Math.min((t-lastT)/1000 || 0.016, 0.05); lastT = t;
  if (playing){
    f += dt*A.fs*rate;
    if (f >= A.F-1){ f = A.F-1; stopPlay(); }
    needDraw = true;
  } else if (!dragScrub){
    if (Math.abs(vF) > 0.5){ applyDelta(vF*dt); vF *= Math.exp(-dt*3.5); } else vF = 0;
    if (f < 0 || f > A.F-1){                  // the watch-spring bounce-back
      const b = clamp(f);
      f += (b-f)*Math.min(1, dt*9);
      if (Math.abs(f-b) < 0.02){ f = b; vF = 0; }
      needDraw = true;
    }
  }
  const k = 1 - Math.exp(-dt*22);             // display chases the target, never jumps
  if (Math.abs(f-fd) > 1e-3){ fd += (f-fd)*k; needDraw = true; } else fd = f;
  if (needDraw){ needDraw = false; drawStage(); }
}
function togglePlay(){ playing ? stopPlay() : startPlay(); }
function startPlay(){ if (f>=A.F-1) f=0;
  playing = true; playBtn.innerHTML = "&#9646;&#9646;"; }
function stopPlay(){ playing = false; playBtn.innerHTML = "&#9654;"; }
playBtn.onclick = togglePlay;
document.getElementById("rateSel").onchange = e => rate = +e.target.value;

// selectors + segments
const runnerSel = document.getElementById("runnerSel"), ghostSel = document.getElementById("ghostSel");
function fillSelects(){
  runnerSel.innerHTML = R.map((r,i)=>`<option value="${i}">${r.id}  ${r.top.toFixed(2)} m/s ${r.sex}</option>`).join("");
  ghostSel.innerHTML = `<option value="-1">no one</option>` +
    R.map((r,i)=> A===r ? "" : `<option value="${i}">${r.id}  ${r.top.toFixed(2)} m/s ${r.sex}</option>`).join("");
}
function setRunner(i){
  const frac = clamp(fd)/(A.F-1);
  A = R[i]; if (B===A) B = null;
  f = fd = clamp(frac*(A.F-1)); vF = 0;
  document.getElementById("runnerInfo").textContent =
    `${A.id}: ${A.top.toFixed(2)} m/s top speed, ${A.steps60} steps to 60 m, ` +
    `${A.stride_m.toFixed(2)} m at ${A.stride_hz.toFixed(2)} Hz`;
  fillSelects(); runnerSel.value = i; ghostSel.value = B ? R.indexOf(B) : -1;
  refresh();
}
runnerSel.onchange = e => setRunner(+e.target.value);
ghostSel.onchange = e => { B = +e.target.value<0 ? null : R[+e.target.value]; syncSegs(); refresh(); };

function placePill(el){
  const on = el.querySelector("button.on"), pill = el.querySelector(".pill");
  pill.style.width = on.offsetWidth + "px";
  pill.style.transform = `translateX(${on.offsetLeft}px)`;
}
function segInit(id, attr, set){
  const el = document.getElementById(id);
  el.querySelectorAll("button").forEach(b => b.onclick = () => {
    el.querySelectorAll("button").forEach(x=>x.classList.toggle("on", x===b));
    placePill(el);
    set(b.dataset[attr]); refresh();
  });
}
segInit("viewSeg","v", v => { view = v; syncSegs(); });
segInit("modeSeg","m", m => mode = m);
segInit("alignSeg","a", a => align = a);
function syncSegs(){
  document.getElementById("modeSeg").classList.toggle("off", !B || view==="front");
  document.getElementById("alignSeg").classList.toggle("off", !B);
  document.getElementById("stageWrap").classList.toggle("threed", is3D());
  document.getElementById("hint").innerHTML = is3D()
    ? `drag rotates, wheel zooms, timeline scrubs, <kbd>&larr;</kbd><kbd>&rarr;</kbd> step one frame, <kbd>space</kbd> plays`
    : `drag the stage or the timeline to scrub, <kbd>&larr;</kbd><kbd>&rarr;</kbd> step one frame, <kbd>space</kbd> plays`;
}
const placePills = () => document.querySelectorAll(".seg").forEach(placePill);
function refresh(){ drawScrubBase(); needDraw = true; }

addEventListener("resize", () => { placePills(); refresh(); });
document.fonts.ready.then(placePills);
placePills();
velWrap.style.width = velW+"px";
setRunner(0);
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build()
