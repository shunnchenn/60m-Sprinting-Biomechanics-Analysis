"""stride.py — when each foot is on the ground, and the kinogram events built on that.

Input is always one cleaned trial from `sprint_pipeline.clean_for_angles`: frame 0 is the
set position (hands still down), X = forward, Z = up (gravity, from the IMU).

Every event is a frame index into that cleaned trial. At 60 Hz one frame is 16.7 ms, so
each event is good to about one frame.

Order of the file follows the run:
    1. Ground contact      — the one rule everything else is built on
    2. Block start         — set, rear clearance, rear ankle cross, front clearance
    3. Acceleration steps  — touchdown, ankle cross, take-off, until top speed
    4. Top-speed kinogram  — toe-off, max vertical projection, strike, touchdown, full support
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import percentile_filter

import sprint_pipeline as SP

# ── Landmarks (columns of raw64) ─────────────────────────────────────────────
FOOT  = {"R": [SP.MK_R_HEEL, SP.MK_R_TOE], "L": [SP.MK_L_HEEL, SP.MK_L_TOE]}
ANKLE = {"R": SP.MK_R_ANKLE, "L": SP.MK_L_ANKLE}      # lateral + medial malleolus
KNEE  = {"R": SP.MK_R_KNEE,  "L": SP.MK_L_KNEE}       # lateral + medial epicondyle
HIP   = {"R": SP.MK_R_HIP,   "L": SP.MK_L_HIP}        # greater trochanter
ASIS  = [1, 2]                                         # pRightASI, pLeftASI: front of the pelvis
OTHER = {"R": "L", "L": "R"}
X, Z = SP._AP, SP._VERT

# ── The contact rule. Both thresholds are the ones sprint_report.py already uses. ──
BAND_M      = SP._CONTACT_BAND_M   # 0.03: lowest foot point within 3 cm of the local floor
STILL_MS    = 1.5                  # = sprint_report.CONTACT_SPEED_MS: lowest point moving < 1.5 m/s
FLOOR_WIN_S = 1.0                  # local floor over ±0.5 s (the floor drifts 3-5 cm over 60 m)
FLOOR_PCT   = 10                   # 10th percentile, not the minimum: the toe dips 3 cm at every push-off
MIN_FRAMES  = 2                    # a contact shorter than 2 frames (33 ms) is noise, not a step
MAX_GAP     = 5                    # same-foot pieces <= 5 frames (83 ms) apart are one contact


def _at(raw64, cols, axis):
    """Mean of one or more landmarks along one axis, per frame."""
    return raw64[:, np.atleast_1d(cols), axis].mean(axis=1)


def _runs(mask):
    """(start, end) of every run of True, both ends inclusive."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))


# ==========================================================================
# 1 · Ground contact
# ==========================================================================

@dataclass
class Contact:
    """One foot on the ground: first frame down (touchdown) to last frame down (take-off)."""
    foot: str
    touchdown: int
    take_off: int          # = toe-off: the last frame the foot is still on the ground
    on_blocks: bool = False

    @property
    def frames(self):
        return self.take_off - self.touchdown + 1


def foot_contacts(raw64, fs, foot):
    """Every contact of one foot, in time order. The first may be the block pedal.

    A foot is DOWN when its lower point (heel or toe) is near the local floor AND barely
    moving. Height alone misfires when the floor drifts; speed alone misfires on a swing
    foot passing slowly near the ground. Together they agree with how Xsens MVN models
    the foot: pinned still while it is in contact.

    The block pedals are above the floor, so the height test cannot see them. In the set
    position the foot is simply still, so the block contact is the still run from frame 0.
    """
    heel_toe = raw64[:, FOOT[foot], :]
    low   = heel_toe[:, :, Z].min(axis=1)
    speed = np.linalg.norm(np.gradient(heel_toe[:, :, :2], 1 / fs, axis=0), axis=2).min(axis=1)
    still = speed < STILL_MS
    floor = percentile_filter(low, FLOOR_PCT, size=int(FLOOR_WIN_S * fs) | 1)

    found = []
    if still[0]:                                         # 1. on the blocks at the set position
        a, b = _runs(still)[0]
        found.append(Contact(foot, int(a), int(b), on_blocks=True))
    after = found[0].take_off if found else -1
    for a, b in _runs(still & (low - floor < BAND_M)):   # 2. every ground contact after that
        if a > after:
            found.append(Contact(foot, int(a), int(b)))
    return found


def _join(contacts):
    """Join back-to-back pieces of the same foot that are at most MAX_GAP frames apart."""
    out = []
    for c in contacts:
        prev = out[-1] if out else None
        if (prev and prev.foot == c.foot and not (prev.on_blocks or c.on_blocks)
                and c.touchdown - prev.take_off <= MAX_GAP):
            prev.take_off = c.take_off
        else:
            out.append(c)
    return out


def all_contacts(raw64, fs):
    """Both feet, ordered by touchdown, one entry per real contact.

    Three clean-up rules, each a physical fact about a block start and sprinting:
      1. Nothing touches the track until both feet have left the blocks.
      2. The Xsens foot model sometimes slides for a few frames, or rocks heel to toe by
         more than BAND_M, which breaks one contact into pieces. A foot cannot land twice
         within one flight phase, so same-foot pieces <= MAX_GAP frames apart are one contact.
         (A wider gap means the other foot's contact was missed: those are NOT joined.)
      3. A contact shorter than MIN_FRAMES is noise.
    """
    found = sorted(foot_contacts(raw64, fs, "R") + foot_contacts(raw64, fs, "L"),
                   key=lambda c: c.touchdown)
    left_blocks = max((c.take_off for c in found if c.on_blocks), default=-1)
    found = [c for c in found if c.on_blocks or c.touchdown > left_blocks]     # rule 1
    found = [c for c in _join(found) if c.on_blocks or c.frames >= MIN_FRAMES]  # rules 2, 3
    return _join(found)                                                         # rule 2 again


def ankle_cross(raw64, swing, stance, start, end):
    """First frame in [start, end] where the swing ankle passes the stance KNEE (Kinogram Notes)."""
    ahead = _at(raw64, ANKLE[swing], X) >= _at(raw64, KNEE[stance], X)
    hits = np.where(ahead[start:end + 1])[0]
    return int(start + hits[0]) if len(hits) else None


# ==========================================================================
# 2 · Block start
# ==========================================================================

@dataclass
class BlockStart:
    """Kinogram Notes block events. Clearance = the foot's last frame on its pedal."""
    rear_foot: str
    front_foot: str
    set_position: int        # frame 0: hands still down
    rear_clearance: int
    rear_ankle_cross: int | None
    front_clearance: int     # = block take-off


def block_start(raw64, contacts):
    """Rear foot = the one further back at the set position.

    Rear ankle cross is searched until step 1 lands: some athletes cross after the front
    foot has already left its pedal.
    """
    blocks = {c.foot: c for c in contacts if c.on_blocks}
    if len(blocks) < 2:
        raise ValueError("both feet must be still at frame 0 (set position)")
    rear = "R" if raw64[0, SP.MK_R_HEEL, X] < raw64[0, SP.MK_L_HEEL, X] else "L"
    front = OTHER[rear]
    rc, fc = blocks[rear].take_off, blocks[front].take_off
    step1 = next((c.touchdown for c in contacts if not c.on_blocks), len(raw64) - 1)
    return BlockStart(rear, front, 0, rc, ankle_cross(raw64, rear, front, rc, step1), fc)


# ==========================================================================
# 3 · Acceleration: every step from the blocks to top speed
# ==========================================================================

@dataclass
class Step:
    """One step: a ground contact, the other ankle passing that knee, then take-off."""
    number: int
    foot: str
    touchdown: int
    ankle_cross: int | None
    take_off: int


def acceleration_steps(raw64, contacts, until_frame):
    """Steps 1, 2, ... after the blocks, up to the step in contact at `until_frame`.

    Ankle cross is searched from touchdown to the NEXT touchdown: in the first steps
    the swing foot can still be behind the knee when the stance foot leaves the ground.
    """
    ground = [c for c in contacts if not c.on_blocks]
    steps = []
    for i, c in enumerate(ground):
        if c.touchdown > until_frame:
            break
        nxt = ground[i + 1].touchdown if i + 1 < len(ground) else len(raw64) - 1
        steps.append(Step(i + 1, c.foot, c.touchdown,
                          ankle_cross(raw64, OTHER[c.foot], c.foot, c.touchdown, nxt), c.take_off))
    return steps


# ==========================================================================
# 4 · Top-speed kinogram (ALTIS five positions)
# ==========================================================================

@dataclass
class KinogramStep:
    """ALTIS kinogram for one step: off one foot, onto the other.

    toe_off      last frame the pushing foot is on the ground
    max_vertical highest pelvis point in the flight that follows
    strike       the pushing leg's thigh closest to vertical, in flight (before touchdown)
    touchdown    first frame the landing foot is on the ground
    full_support first frame the landing toe is under the front of the pelvis (ASIS)
    """
    off_foot: str
    on_foot: str
    toe_off: int
    max_vertical: int
    strike: int
    touchdown: int
    full_support: int | None


def thigh_from_vertical(raw64, foot):
    """Thigh angle from vertical in degrees, per frame (0 = hanging straight down)."""
    dx = _at(raw64, KNEE[foot], X) - _at(raw64, HIP[foot], X)
    dz = _at(raw64, KNEE[foot], Z) - _at(raw64, HIP[foot], Z)
    return np.degrees(np.arctan2(dx, -dz))


def kinogram_step(raw64, push, land):
    """The five positions between contact `push` and the next contact `land`."""
    a, b = push.take_off, land.touchdown
    pelvis_z = _at(raw64, SP.MK_PELVIS_ALL, Z)
    mvp = a + int(np.argmax(pelvis_z[a:b + 1]))
    end = max(mvp, b - 1)                                       # strike is in flight, before touchdown
    strike = mvp + int(np.argmin(np.abs(thigh_from_vertical(raw64, push.foot)[mvp:end + 1])))
    under = _at(raw64, ASIS, X) >= _at(raw64, FOOT[land.foot][1], X)    # ASIS has passed the toe
    hits = np.where(under[land.touchdown:land.take_off + 1])[0]
    fs_ = land.touchdown + int(hits[0]) if len(hits) else None
    return KinogramStep(push.foot, land.foot, a, mvp, strike, b, fs_)


def top_speed_kinogram(raw64, contacts, peak_frame, n_steps=2):
    """`n_steps` kinogram steps, starting at the first toe-off at or after top speed.

    Two steps = one full stride (right and left), which is the default.
    """
    ground = [c for c in contacts if not c.on_blocks]
    first = next((i for i, c in enumerate(ground) if c.take_off >= peak_frame), None)
    if first is None or first + 1 >= len(ground):
        raise ValueError(f"no complete step after top speed (frame {peak_frame}); "
                         "the runner may still be accelerating at the end of the capture")
    return [kinogram_step(raw64, ground[i], ground[i + 1])
            for i in range(first, min(first + n_steps, len(ground) - 1))]


if __name__ == "__main__":
    # Self-check on SB80: events must come out in physical order.
    t = SP.clean_for_angles("SB80")
    raw, fs = t["raw64"], t["fs"]
    cs = all_contacts(raw, fs)
    blk = block_start(raw, cs)
    assert blk.set_position < blk.rear_clearance < blk.front_clearance
    for k in top_speed_kinogram(raw, cs, t["peak_frame"]):
        assert k.toe_off < k.max_vertical <= k.strike < k.touchdown <= (k.full_support or k.touchdown)
    print("stride.py self-check passed on SB80")
