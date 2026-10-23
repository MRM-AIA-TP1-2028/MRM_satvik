"""v4: v1's detections + a guided search in the gaps + v3's clean-up.

Why: a diagnosis of v1's misses (v3/output/missed_ball_examples.jpg) showed:
  - ~half the time the ball is not in the image at all (motion blur erases a
    fast 4-px ball, or it crosses a white line) -> only interpolation helps;
  - but often it IS visible and v1 throws it away: the blob is so small the
    2x2 noise-removal step deletes it, the motion is just under the
    threshold, or it is next to a player at a hit (v1 excludes those areas).

v1's strict rules are needed across the whole frame, otherwise crowd, feet and
line shimmer get in. But right where a track says the ball must be, a false
blob is unlikely, so v4 relaxes the rules only there:

  1. Strict pass  - v1's detector, unchanged (same detections as v1/v3).
  2. Guided pass  - after the fact, follow each v1 run forwards from its end
                    and backwards from its start along a curve fitted to the
                    run, and search a small window at each predicted position
                    with relaxed rules: no noise-removal step, lower motion
                    and brightness thresholds, players allowed (still inside
                    the play area). v1's runs are never changed, so the
                    guided pass can only ADD points next to existing tracks.
  3. v3 clean-up  - extend runs onto on-path candidates, drop blips, fill
                    short gaps whose flights line up.

Usage:
    conda activate SANDBOX-MAX
    python track_ball_v4.py
    python track_ball_v4.py --no-cache     # rerun v1's detector

Output: v4/output/<input name>_tracked_v4.mp4
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
sys.path[:0] = [TASK_DIR, os.path.join(TASK_DIR, "v1"), os.path.join(TASK_DIR, "v3")]
import ball_and_blue as bb  # noqa: E402
import track_ball_v3 as v3  # noqa: E402
from test_ball_on_video import BroadcastDetector  # noqa: E402

DEFAULT_INPUT = os.path.join(TASK_DIR, "videos", "test_video.mp4")
OUTPUT_DIR = os.path.join(HERE, "output")
V1_CACHE = os.path.join(TASK_DIR, "v3", "output", "{name}_v1_detections.json")

# Which runs are followed, and how far.
MIN_RUN_POINTS = 3           # only runs this long are trusted as anchors
FIT_POINTS = 6               # points used to fit the run's curve
MAX_GUIDED = 6               # frames searched beyond each end of a run
MAX_GUIDED_MISSES = 1        # empty windows allowed before following stops

# Relaxed rules, used only inside the window at each predicted position.
GUIDED_RADIUS, GUIDED_GROWTH = 8, 3     # px window (+ per frame from the run)
GUIDED_MOTION = 14                      # v1: 25, and no 2x2 noise removal
GUIDED_TOPHAT = 20                      # v1: 30
GUIDED_SATURATION = 170                 # v1: 150
GUIDED_MAX_AREA = 120
GUIDED_MAX_ASPECT = 4.0
TOPHAT = BroadcastDetector.TOPHAT


# --------------------------------------------------------------------------
# Predicting where the ball should be next to a run
# --------------------------------------------------------------------------
def predictor(run, forward):
    """Fit x(t), y(t) to the run's end (or start). Quadratic when there are
    enough points, so the arc of a lob is followed; linear otherwise."""
    pts = run[-FIT_POINTS:] if forward else run[:FIT_POINTS]
    t = np.array([p[0] for p in pts], float)
    deg = 2 if len(pts) >= 5 else 1
    fx = np.polyfit(t, [p[1] for p in pts], deg)
    fy = np.polyfit(t, [p[2] for p in pts], deg)
    return lambda f: (float(np.polyval(fx, f)), float(np.polyval(fy, f)))


def search_windows(dets):
    """For every frame, the windows to search: {frame: [(key, k, pred, radius)]}."""
    windows = defaultdict(list)
    runs = v3.split_runs(dets)
    for i, run in enumerate(runs):
        if len(run) < MIN_RUN_POINTS:
            continue
        for forward in (True, False):
            pred = predictor(run, forward)
            edge = run[-1][0] if forward else run[0][0]
            for k in range(1, MAX_GUIDED + 1):
                f = edge + k if forward else edge - k
                if not 0 <= f < len(dets) or dets[f]:
                    break                      # reached the video edge or another run
                windows[f].append(((i, forward), k, pred(f), GUIDED_RADIUS + GUIDED_GROWTH * k))
    return windows


# --------------------------------------------------------------------------
# The relaxed search inside one window
# --------------------------------------------------------------------------
def guided_search(frame, dmin, roi, pred, radius):
    """Best ball-like blob within radius of pred under relaxed rules, or None."""
    h, w = frame.shape[:2]
    pad = 6                                    # room for the top-hat kernel
    x0, y0 = int(pred[0] - radius - pad), int(pred[1] - radius - pad)
    x1, y1 = int(pred[0] + radius + pad) + 1, int(pred[1] + radius + pad) + 1
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)

    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, TOPHAT)
    bright = cv2.bitwise_and(cv2.inRange(tophat, GUIDED_TOPHAT, 255),
                             cv2.inRange(hsv[..., 1], 0, GUIDED_SATURATION))
    lo, hi = bb.HSV_RANGES["ball"]
    bright = cv2.bitwise_or(bright, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    moving = cv2.inRange(dmin[y0:y1, x0:x1], GUIDED_MOTION, 255)
    mask = cv2.bitwise_and(cv2.bitwise_and(bright, moving), roi[y0:y1, x0:x1])
    mask = cv2.dilate(mask, None, iterations=1)

    best, best_d = None, None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        _, _, bw, bh = cv2.boundingRect(c)
        if bw * bh > GUIDED_MAX_AREA or max(bw, bh) / max(min(bw, bh), 1) > GUIDED_MAX_ASPECT:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        centre = (int(cx) + x0, int(cy) + y0)
        d = math.dist(centre, pred)
        if d <= radius and (best is None or d < best_d):
            best, best_d = (centre, max(r, 1.5)), d
    return best


def guided_pass(path, dets):
    """Search every window in one pass over the video; returns new detections."""
    windows = search_windows(dets)
    found = {}                                  # (key, k) -> (frame, x, y, r)
    cap = cv2.VideoCapture(path)
    history = deque(maxlen=2)
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if f in windows and len(history) == 2:
            dmin = cv2.min(cv2.absdiff(grey, history[-1]), cv2.absdiff(grey, history[-2]))
            roi = BroadcastDetector.play_area(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
            for key, k, pred, radius in windows[f]:
                hit = guided_search(frame, dmin, roi, pred, radius)
                if hit:
                    found[(key, k)] = (f, hit[0][0], hit[0][1], round(hit[1], 1))
        history.append(grey)
        f += 1
    cap.release()

    # Follow each run outwards; stop after too many empty windows in a row.
    added = {}
    for key in {key for wl in windows.values() for key, *_ in wl}:
        misses = 0
        for k in range(1, MAX_GUIDED + 1):
            hit = found.get((key, k))
            if hit is None:
                misses += 1
                if misses > MAX_GUIDED_MISSES:
                    break
                continue
            misses = 0
            if hit[0] not in added:             # forward and backward may meet
                added[hit[0]] = list(hit[1:])
    return added


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", help="default: v4/output/<name>_tracked_v4.mp4")
    ap.add_argument("--no-cache", action="store_true", help="rerun v1's detector")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or os.path.join(OUTPUT_DIR, f"{name}_tracked_v4.mp4")
    cache = V1_CACHE.format(name=name)

    start = time.time()
    if os.path.isfile(cache) and not args.no_cache:
        dets, cands = json.load(open(cache))
        print(f"loaded v1 detections from {cache}")
    else:
        dets, cands = v3.run_v1(args.input)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump([dets, cands], open(cache, "w"))

    added = guided_pass(args.input, dets)
    combined = list(dets)
    for f, d in added.items():
        combined[f] = d
    json.dump(sorted(added), open(os.path.join(OUTPUT_DIR, f"{name}_guided_frames.json"), "w"))

    pos, extended, dropped = v3.clean(combined, cands)
    n_v1 = sum(1 for d in dets if d)
    n_det = sum(1 for p in pos if p and p[3] == v3.DETECTED)
    n_int = sum(1 for p in pos if p and p[3] == v3.INTERPOLATED)
    n_guided = sum(1 for f in added if pos[f] and pos[f][3] == v3.DETECTED)
    print(f"v1: {n_v1} frames | guided search added {len(added)}")
    print(f"v4: {n_det} detected ({n_guided} guided, +{extended} extended, "
          f"-{dropped} blips) + {n_int} interpolated = {n_det + n_int}/{len(pos)} "
          f"({100 * (n_det + n_int) / len(pos):.1f}%)")

    v3.render(args.input, out_path, pos, label="v4")
    print(f"saved {out_path} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
