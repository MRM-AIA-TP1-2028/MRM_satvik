"""v2: offline tennis ball tracking on a broadcast video.

v1 tracked frame by frame and kept only one hypothesis: when the ball dropped
out for a few frames, or changed direction at a hit or bounce, the track died
and needed three fresh frames to restart. Most of its misses were these short
gaps. Because the input is a saved video, v2 can look at the whole clip:

  1. Candidates   - per frame, small bright moving blobs inside the play area
                    (same cues as v1 with looser thresholds).
  2. Tracklets    - many hypotheses followed at once with a constant-velocity
                    model, so a false track cannot steal the real ball.
  3. Filtering    - keep tracklets that move like a ball: long enough, really
                    moving, smooth, and not mostly on a player.
  4. Selection    - choose the best non-overlapping set of tracklets over the
                    whole clip (weighted interval scheduling).
  5. Gap filling  - neighbouring tracklets are bridged by interpolation when
                    they belong to one ball path: for very short gaps (a hit
                    or bounce) the distance must be reachable at the ball's
                    speed; for longer gaps one tracklet's straight-line
                    flight must land on the other.

Detected and interpolated positions are drawn differently and counted
separately.

Usage:
    conda activate SANDBOX-MAX
    python track_ball_v2.py
    python track_ball_v2.py --input ../videos/other.mp4 --no-cache

Output: v2/output/<input name>_tracked_v2.mp4
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
sys.path.insert(0, TASK_DIR)
import ball_and_blue as bb  # noqa: E402

DEFAULT_INPUT = os.path.join(TASK_DIR, "videos", "test_video.mp4")
OUTPUT_DIR = os.path.join(HERE, "output")

# --- candidates -----------------------------------------------------------
MOTION_THRESH = 18          # grey-level change vs both previous frames (v1: 25)
TOPHAT_THRESH = 22          # brighter than surroundings by this much (v1: 30)
MAX_SATURATION = 150
MIN_AREA, MAX_AREA = 2, 150
MAX_ASPECT = 3.5
PLAYER_AREA = 1500          # merged moving blob bigger than this = person
SCENE_CUT = 0.25
MIN_ROI = 0.2
TOPHAT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
MERGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# --- tracklets ------------------------------------------------------------
MAX_STEP = 70               # px/frame, fastest plausible ball
GATE, GATE_GROWTH = 18, 10  # search radius around the prediction
MAX_MISSED = 4              # frames a tracklet may coast without a match

# --- filtering ------------------------------------------------------------
MIN_POINTS = 5
MIN_TRAVEL = 25             # px between first and last point
MIN_MEDIAN_STEP = 3.0       # px/frame, slower movers are people
MAX_MEDIAN_ACCEL = 6.0      # px/frame^2, ball paths are smooth
MAX_NEAR_PLAYER = 0.6       # fraction of points allowed next to a player

# --- gap filling ----------------------------------------------------------
MAX_GAP = 12                # frames
GAP_REACH = 20              # px allowed on top of what the ball's own speed covers
GAP_SPEED_SLACK = 1.5       # the ball may speed up this much across a gap
HIT_GAP = 3                 # gaps up to this long may change direction (hit/bounce)
PREDICT_TOL, PREDICT_TOL_GROWTH = 15, 3   # px miss allowed for longer gaps

DETECTED, INTERPOLATED = 1, 2


# --------------------------------------------------------------------------
# 1. Candidates
# --------------------------------------------------------------------------
def play_area(hsv):
    """Court (blue) + surround (green), filled in and shrunk. Excludes the crowd."""
    surface = cv2.bitwise_or(cv2.inRange(hsv, (45, 40, 80), (70, 140, 200)),
                             cv2.inRange(hsv, (98, 35, 90), (118, 140, 255)))
    surface = cv2.morphologyEx(surface, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    roi = np.zeros_like(surface)
    contours, _ = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(roi, [cv2.convexHull(max(contours, key=cv2.contourArea))],
                         -1, 255, -1)
    return cv2.erode(roi, np.ones((31, 31), np.uint8))


def frame_candidates(frame, history):
    """Return [(x, y, r, near_player)] for one frame, or None on a scene cut."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if len(history) < 2:
        history.append(grey)
        return []
    moving = cv2.inRange(cv2.min(cv2.absdiff(grey, history[-1]),
                                 cv2.absdiff(grey, history[-2])), MOTION_THRESH, 255)
    history.append(grey)
    moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if cv2.countNonZero(moving) > SCENE_CUT * moving.size:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = play_area(hsv)
    if cv2.countNonZero(roi) < MIN_ROI * roi.size:
        return []

    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, TOPHAT)
    bright = cv2.bitwise_and(cv2.inRange(tophat, TOPHAT_THRESH, 255),
                             cv2.inRange(hsv[..., 1], 0, MAX_SATURATION))
    lo, hi = bb.HSV_RANGES["ball"]
    yellow = cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.bitwise_and(moving, cv2.bitwise_or(bright, yellow))
    mask = cv2.dilate(cv2.bitwise_and(mask, roi), None, iterations=1)

    # Players are flagged rather than removed: the ball is near a player at
    # every hit, and the tracklet filter decides later.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(moving, MERGE))
    players = np.isin(labels, [k for k in range(1, n)
                               if stats[k, cv2.CC_STAT_AREA] > PLAYER_AREA])

    cands = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not MIN_AREA <= max(cv2.contourArea(c), w * h / 2) <= MAX_AREA:
            continue
        if max(w, h) / max(min(w, h), 1) > MAX_ASPECT:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        cands.append((round(cx, 1), round(cy, 1), round(max(r, 1.5), 1),
                      bool(players[int(cy), int(cx)])))
    return cands


def collect_candidates(path):
    cap = cv2.VideoCapture(path)
    history, frames = deque(maxlen=2), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cands = frame_candidates(frame, history)
        frames.append(cands if cands is not None else "cut")
        if len(frames) % 500 == 0:
            print(f"  candidates: {len(frames)} frames")
    cap.release()
    return frames


# --------------------------------------------------------------------------
# 2. Tracklets
# --------------------------------------------------------------------------
def build_tracklets(frames):
    """Multi-hypothesis constant-velocity linking. Returns lists of points
    (frame, x, y, r, near_player)."""
    active, done = [], []

    def finish(t):
        if len(t["pts"]) >= 3:
            done.append(t["pts"])

    for f, cands in enumerate(frames):
        if cands == "cut":
            for t in active:
                finish(t)
            active = []
            continue

        pairs = []
        for ti, t in enumerate(active):
            last = t["pts"][-1]
            k = f - last[0]
            if t["vel"] is None:                         # one point so far
                pred, gate = last[1:3], MAX_STEP
            else:
                pred = (last[1] + t["vel"][0] * k, last[2] + t["vel"][1] * k)
                gate = GATE + GATE_GROWTH * (k - 1) + 0.2 * math.hypot(*t["vel"])
            for ci, c in enumerate(cands):
                d = math.dist(pred, c[:2])
                if d <= gate:
                    pairs.append((d, ti, ci))

        used_t, used_c = set(), set()
        for d, ti, ci in sorted(pairs):
            if ti in used_t or ci in used_c:
                continue
            t, c = active[ti], cands[ci]
            last = t["pts"][-1]
            k = f - last[0]
            t["vel"] = ((c[0] - last[1]) / k, (c[1] - last[2]) / k)
            t["pts"].append((f, *c))
            used_t.add(ti)
            used_c.add(ci)

        still = []
        for ti, t in enumerate(active):
            if ti in used_t or f - t["pts"][-1][0] <= MAX_MISSED:
                still.append(t)
            else:
                finish(t)
        active = still + [{"pts": [(f, *c)], "vel": None}
                          for ci, c in enumerate(cands) if ci not in used_c]

    for t in active:
        finish(t)
    return done


# --------------------------------------------------------------------------
# 3. Filtering
# --------------------------------------------------------------------------
def tracklet_quality(pts):
    """Score > 0 for ball-like tracklets, 0 otherwise."""
    if len(pts) < MIN_POINTS:
        return 0
    f = np.array([p[0] for p in pts], float)
    xy = np.array([p[1:3] for p in pts], float)
    if np.linalg.norm(xy[-1] - xy[0]) < MIN_TRAVEL:
        return 0
    vel = np.diff(xy, axis=0) / np.diff(f)[:, None]
    if np.median(np.linalg.norm(vel, axis=1)) < MIN_MEDIAN_STEP:
        return 0
    if len(vel) >= 2:
        accel = np.diff(vel, axis=0)
        if np.median(np.linalg.norm(accel, axis=1)) > MAX_MEDIAN_ACCEL:
            return 0
    near = np.mean([p[4] for p in pts])
    if near > MAX_NEAR_PLAYER:
        return 0
    return len(pts) * (1 - 0.5 * near)


# --------------------------------------------------------------------------
# 4. Selection: best non-overlapping set (weighted interval scheduling)
# --------------------------------------------------------------------------
def select_tracklets(tracklets):
    scored = [(t, tracklet_quality(t)) for t in tracklets]
    scored = sorted([s for s in scored if s[1] > 0], key=lambda s: s[0][-1][0])
    ends = [t[-1][0] for t, _ in scored]
    best = [0.0] * (len(scored) + 1)
    take = [False] * len(scored)
    prev = []
    for i, (t, score) in enumerate(scored):
        # last tracklet that ends before this one starts
        j = int(np.searchsorted(ends, t[0][0], side="left"))
        prev.append(j)
        if score + best[j] > best[i]:
            best[i + 1], take[i] = score + best[j], True
        else:
            best[i + 1] = best[i]
    chosen, i = [], len(scored)
    while i > 0:
        if take[i - 1]:
            chosen.append(scored[i - 1][0])
            i = prev[i - 1]
        else:
            i -= 1
    return chosen[::-1]


# --------------------------------------------------------------------------
# 5. Per-frame positions with gap filling
# --------------------------------------------------------------------------
def positions(chosen, n_frames):
    """Return per-frame (x, y, r, kind) or None."""
    out = [None] * n_frames

    def velocity(t, at_end):
        """Ball velocity (px/frame) over the last/first few points of a tracklet."""
        seg = t[-4:] if at_end else t[:4]
        k = max(seg[-1][0] - seg[0][0], 1)
        return ((seg[-1][1] - seg[0][1]) / k, (seg[-1][2] - seg[0][2]) / k)

    def fill(a, b):
        gap = b[0] - a[0]
        for k in range(1, gap):
            s = k / gap
            out[a[0] + k] = (a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2]),
                             a[3] + s * (b[3] - a[3]), INTERPOLATED)

    def joins(ta, tb):
        """Do the end of tracklet ta and the start of tb belong to one ball path?"""
        a, b = ta[-1], tb[0]
        gap = b[0] - a[0]
        if gap <= 1 or gap > MAX_GAP + 1:
            return False
        va, vb = velocity(ta, True), velocity(tb, False)
        if gap <= HIT_GAP:
            # A hit or bounce: direction may change, distance must be reachable.
            speed = max(math.hypot(*va), math.hypot(*vb))
            return math.dist(a[1:3], b[1:3]) <= GAP_REACH + GAP_SPEED_SLACK * speed * gap
        # Longer gap: one side's straight-line flight must land on the other.
        fwd = (a[1] + va[0] * gap, a[2] + va[1] * gap)
        back = (b[1] - vb[0] * gap, b[2] - vb[1] * gap)
        tol = PREDICT_TOL + PREDICT_TOL_GROWTH * gap
        return min(math.dist(fwd, b[1:3]), math.dist(back, a[1:3])) <= tol

    for i, t in enumerate(chosen):
        for p in t:
            out[p[0]] = (p[1], p[2], p[3], DETECTED)
        for a, b in zip(t, t[1:]):                       # coasting inside a tracklet
            fill(a, b)
        if i + 1 < len(chosen) and joins(t, chosen[i + 1]):
            fill(t[-1], chosen[i + 1][0])                # between tracklets
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render(path, out_path, pos):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail = deque(maxlen=30)
    n_det = n_int = 0
    for f, p in enumerate(pos):
        ok, frame = cap.read()
        if not ok:
            break
        if p:
            x, y, r, kind = p
            c = (int(round(x)), int(round(y)))
            ball = {"centre": c, "radius": r, "area": math.pi * r * r}
            if kind == DETECTED:
                n_det += 1
                bb.draw_ball(frame, ball)
                cv2.circle(frame, c, int(r) + 10, bb.BALL, 1)
            else:
                n_int += 1
                cv2.circle(frame, c, int(r) + 10, (200, 200, 200), 1)
                bb.put_text(frame, "ball (interpolated)", (c[0] - 40, c[1] - 16),
                            (200, 200, 200), 0.45)
            trail.appendleft(c)
        else:
            trail.appendleft(None)
        for i in range(1, len(trail)):
            if trail[i - 1] and trail[i]:
                thickness = int(math.sqrt(30 / float(i + 1)) * 1.5) + 1
                cv2.line(frame, trail[i - 1], trail[i], (0, 140, 255), thickness)
        seen = n_det + n_int
        bb.put_text(frame, f"v2 | frame {f + 1}/{len(pos)} | ball: "
                           f"{'detected' if p and p[3] == DETECTED else 'interpolated' if p else 'no'}"
                           f" | tracked {100 * seen / (f + 1):.0f}% of frames so far",
                    (10, 22), bb.WHITE, 0.55)
        writer.write(frame)
    cap.release()
    writer.release()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", help="default: v2/output/<name>_tracked_v2.mp4")
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute candidates instead of reusing the cached ones")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or os.path.join(OUTPUT_DIR, f"{name}_tracked_v2.mp4")
    cache = os.path.join(OUTPUT_DIR, f"{name}_candidates.json")

    start = time.time()
    if os.path.isfile(cache) and not args.no_cache:
        frames = json.load(open(cache))
        print(f"loaded candidates from {cache}")
    else:
        frames = collect_candidates(args.input)
        json.dump(frames, open(cache, "w"))

    tracklets = build_tracklets(frames)
    chosen = select_tracklets(tracklets)
    pos = positions(chosen, len(frames))
    n_det = sum(1 for p in pos if p and p[3] == DETECTED)
    n_int = sum(1 for p in pos if p and p[3] == INTERPOLATED)
    print(f"{len(tracklets)} tracklets, {len(chosen)} kept")
    print(f"ball tracked in {n_det + n_int}/{len(pos)} frames "
          f"({100 * (n_det + n_int) / len(pos):.1f}%): "
          f"{n_det} detected + {n_int} interpolated")

    render(args.input, out_path, pos)
    print(f"saved {out_path} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
