"""v3: v1's detector plus conservative clean-up. Precision first.

v2 found the ball in more frames but added too many false positives, because
it loosened the detector and filled time with any ball-like tracklet. v3 does
not loosen anything. It runs v1's broadcast detector unchanged, then:

  1. Extends runs - v1 never draws the two frames that confirm a new track,
                    and gives up at the end of a run. Each run is extended
                    forwards/backwards onto v1's own candidates, but only
                    those lying on the run's straight-line path (a few px).
  2. Drops blips  - runs of 1-2 detections that do not line up with any
                    neighbouring run (v1's most likely false hits).
  3. Fills gaps   - only short gaps (<= 5 frames) between two runs where one
                   run's straight-line flight lands on the other. Filled
                   positions are drawn differently and counted separately.

Usage:
    conda activate SANDBOX-MAX
    python track_ball_v3.py
    python track_ball_v3.py --no-cache     # rerun v1's detector

Output: v3/output/<input name>_tracked_v3.mp4
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
sys.path.insert(0, TASK_DIR)
sys.path.insert(0, os.path.join(TASK_DIR, "v1"))
import ball_and_blue as bb  # noqa: E402
from test_ball_on_video import BroadcastDetector  # noqa: E402  (v1, unchanged)

DEFAULT_INPUT = os.path.join(TASK_DIR, "videos", "test_video.mp4")
OUTPUT_DIR = os.path.join(HERE, "output")

MIN_RUN = 3                  # runs shorter than this must line up with a neighbour
LINK_GAP = 8                 # frames: max gap for a blip to count as linked
MAX_FILL_GAP = 5             # frames: longest gap that is interpolated
FIT_POINTS = 4               # points used to estimate a run's velocity
PREDICT_TOL, PREDICT_TOL_GROWTH = 10, 3   # px miss allowed when extrapolating
EXTEND_TOL, EXTEND_TOL_GROWTH = 6, 2      # px: candidate must sit on the path
MAX_EXTEND = 8               # frames a run may grow at each end
EXTEND_MISS = 1              # frames without a candidate before extension stops

DETECTED, INTERPOLATED = 1, 2
TRAIL_MAX_GAP = 2            # frames without the ball before the trail is cleared


def run_v1(path):
    """v1's broadcast detector on every frame.

    Returns (detections, candidates): detections[f] is [x, y, r] or None,
    candidates[f] is v1's filtered candidate list [[x, y, r], ...].
    """
    cap = cv2.VideoCapture(path)
    detector, out, cands = BroadcastDetector(), [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ball, _ = detector.detect(frame)
        out.append([*ball["centre"], round(ball["radius"], 1)] if ball else None)
        latest = detector.cand_hist[-1] if detector.cand_hist else []
        cands.append([[c[0][0], c[0][1], round(c[1], 1)] for c in latest])
        if len(out) % 500 == 0:
            print(f"  v1 detector: {len(out)} frames")
    cap.release()
    return out, cands


def split_runs(dets):
    """Consecutive detections -> list of runs [(frame, x, y, r), ...]."""
    runs, cur = [], []
    for f, d in enumerate(dets):
        if d:
            cur.append((f, *d))
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def velocity(run, at_end):
    seg = run[-FIT_POINTS:] if at_end else run[:FIT_POINTS]
    if len(seg) < 2:
        return None
    k = seg[-1][0] - seg[0][0]
    return ((seg[-1][1] - seg[0][1]) / k, (seg[-1][2] - seg[0][2]) / k)


def lines_up(a, b, max_gap):
    """True if run b continues run a's straight-line flight (either direction)."""
    gap = b[0][0] - a[-1][0]
    if gap < 1 or gap > max_gap + 1:
        return False
    tol = PREDICT_TOL + PREDICT_TOL_GROWTH * gap
    va, vb = velocity(a, True), velocity(b, False)
    end, start = a[-1], b[0]
    if va and math.dist((end[1] + va[0] * gap, end[2] + va[1] * gap), start[1:3]) <= tol:
        return True
    if vb and math.dist((start[1] - vb[0] * gap, start[2] - vb[1] * gap), end[1:3]) <= tol:
        return True
    return False


def extend(run, cands, taken, forward):
    """Grow a run onto v1 candidates that lie on its straight-line path."""
    run = list(run)
    misses = 0
    for _ in range(MAX_EXTEND):
        v = velocity(run, forward)
        if v is None:
            break
        ref = run[-1] if forward else run[0]
        step = 1 if forward else -1
        f = ref[0] + step * (misses + 1)
        if not 0 <= f < len(cands) or f in taken:
            break
        k = misses + 1
        pred = (ref[1] + step * v[0] * k, ref[2] + step * v[1] * k)
        tol = EXTEND_TOL + EXTEND_TOL_GROWTH * (k - 1)
        near = [c for c in cands[f] if math.dist(pred, c[:2]) <= tol]
        if not near:
            misses += 1
            if misses > EXTEND_MISS:
                break
            continue
        c = min(near, key=lambda c: math.dist(pred, c[:2]))
        point = (f, c[0], c[1], c[2])
        run = run + [point] if forward else [point] + run
        taken.add(f)
        misses = 0
    return run


def clean(dets, cands):
    runs = split_runs(dets)

    # 1. Extend runs onto on-path v1 candidates (never into another run's frames).
    taken = {f for r in runs for f, *_ in r}
    before = len(taken)
    runs = [extend(extend(r, cands, taken, True), cands, taken, False) for r in runs]
    runs = split_runs(_flatten(runs, len(dets)))        # extensions may join runs
    extended = len(taken) - before

    # 2. Drop short runs that line up with neither neighbour.
    kept = []
    for i, r in enumerate(runs):
        if len(r) >= MIN_RUN:
            kept.append(r)
            continue
        prev_ok = i > 0 and lines_up(runs[i - 1], r, LINK_GAP)
        next_ok = i + 1 < len(runs) and lines_up(r, runs[i + 1], LINK_GAP)
        if prev_ok or next_ok:
            kept.append(r)

    # 3. Per-frame positions, filling short gaps between runs that line up.
    out = [None] * len(dets)
    for i, r in enumerate(kept):
        for f, x, y, rad in r:
            out[f] = (x, y, rad, DETECTED)
        if i + 1 < len(kept) and lines_up(r, kept[i + 1], MAX_FILL_GAP):
            a, b = r[-1], kept[i + 1][0]
            gap = b[0] - a[0]
            for k in range(1, gap):
                s = k / gap
                out[a[0] + k] = (a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2]),
                                 a[3] + s * (b[3] - a[3]), INTERPOLATED)
    dropped = sum(len(r) for r in runs) - sum(len(r) for r in kept)
    return out, extended, dropped


def _flatten(runs, n):
    dets = [None] * n
    for r in runs:
        for f, x, y, rad in r:
            dets[f] = [x, y, rad]
    return dets


def render(path, out_path, pos, label="v3"):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail = deque(maxlen=30)
    seen = 0
    for f, p in enumerate(pos):
        ok, frame = cap.read()
        if not ok:
            break
        if p:
            seen += 1
            x, y, r, kind = p
            c = (int(round(x)), int(round(y)))
            if kind == DETECTED:
                bb.draw_ball(frame, {"centre": c, "radius": r, "area": math.pi * r * r})
                cv2.circle(frame, c, int(r) + 10, bb.BALL, 1)
            else:
                cv2.circle(frame, c, int(r) + 10, (200, 200, 200), 1)
                bb.put_text(frame, "ball (interpolated)", (c[0] - 40, c[1] - 16),
                            (200, 200, 200), 0.45)
            trail.appendleft(c)
        else:
            trail.appendleft(None)
        # Draw the trail only for the current track: stop at the first gap
        # longer than TRAIL_MAX_GAP frames, so a lost ball leaves no frozen line.
        pts, gap = [], 0
        for c in trail:
            if c is None:
                gap += 1
                if gap > TRAIL_MAX_GAP:
                    break
            else:
                gap = 0
                pts.append(c)
        if trail and trail[0] is None and gap > TRAIL_MAX_GAP:
            pts = []
        for i in range(1, len(pts)):
            thickness = int(math.sqrt(30 / float(i + 1)) * 1.5) + 1
            cv2.line(frame, pts[i - 1], pts[i], (0, 140, 255), thickness)
        state = "detected" if p and p[3] == DETECTED else "interpolated" if p else "no"
        bb.put_text(frame, f"{label} | frame {f + 1}/{len(pos)} | ball: {state} | "
                           f"tracked {100 * seen / (f + 1):.0f}% of frames so far",
                    (10, 22), bb.WHITE, 0.55)
        writer.write(frame)
    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", help="default: v3/output/<name>_tracked_v3.mp4")
    ap.add_argument("--no-cache", action="store_true", help="rerun v1's detector")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or os.path.join(OUTPUT_DIR, f"{name}_tracked_v3.mp4")
    cache = os.path.join(OUTPUT_DIR, f"{name}_v1_detections.json")

    start = time.time()
    if os.path.isfile(cache) and not args.no_cache:
        dets, cands = json.load(open(cache))
        print(f"loaded v1 detections from {cache}")
    else:
        dets, cands = run_v1(args.input)
        json.dump([dets, cands], open(cache, "w"))

    pos, extended, dropped = clean(dets, cands)
    n_v1 = sum(1 for d in dets if d)
    n_det = sum(1 for p in pos if p and p[3] == DETECTED)
    n_int = sum(1 for p in pos if p and p[3] == INTERPOLATED)
    print(f"v1: {n_v1} frames | v3: {n_det} detected (+{extended} extended, "
          f"-{dropped} blips dropped) "
          f"+ {n_int} interpolated = {n_det + n_int}/{len(pos)} "
          f"({100 * (n_det + n_int) / len(pos):.1f}%)")

    render(args.input, out_path, pos)
    print(f"saved {out_path} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
