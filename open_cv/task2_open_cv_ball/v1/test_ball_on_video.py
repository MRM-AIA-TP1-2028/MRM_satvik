"""Run the tennis ball detector on a video file and save the tracking overlay.

Two detector modes:

  original   - the exact detector from ball_and_blue.py (HSV yellow-green mask,
               morphology, circularity check). Built for a close-up ball in
               front of a webcam.
  broadcast  - adapted for TV footage where the ball is only a few px wide.
               Video compression smears the ball's colour into the court
               (on a blue court its hue reads as blue), so colour alone fails.
               Instead: moving (background subtraction) AND bright/low
               saturation OR yellow-green, inside the court area, small, and
               moving at a near-constant velocity over consecutive frames.

Usage:
    conda activate SANDBOX-MAX
    python test_ball_on_video.py                              # broadcast mode
    python test_ball_on_video.py --mode original
    python test_ball_on_video.py --input other.mp4 --show     # live preview too

Output: v1/output/<input name>_tracked_<mode>.mp4
"""

import argparse
import math
import os
import time
from collections import deque

import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
sys.path.insert(0, TASK_DIR)
import ball_and_blue as bb

DEFAULT_INPUT = os.path.join(TASK_DIR, "videos", "test_video.mp4")
OUTPUT_DIR = os.path.join(HERE, "output")

TRAIL_LENGTH = 30
TRAIL = (0, 140, 255)


class OriginalDetector:
    """ball_and_blue.py's detector, unchanged."""

    def detect(self, frame):
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2HSV)
        mask = bb.colour_mask(hsv, "ball")
        return bb.detect_ball(mask), mask


class BroadcastDetector:
    """Small, bright, moving, round blob that moves consistently over frames."""

    MIN_AREA, MAX_AREA = 2, 120
    MAX_ASPECT = 3.0
    MIN_STEP, MAX_STEP = 3, 70
    GATE, GATE_GROWTH = 25, 15
    MAX_COAST = 6
    PLAYER_AREA = 1500
    MERGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    SCENE_CUT = 0.25

    MIN_ROI = 0.2
    TOPHAT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def __init__(self):
        self.history = deque(maxlen=2)
        self.cand_hist = deque(maxlen=2)
        self.pos = self.vel = None
        self.missed = 0

    @staticmethod
    def play_area(hsv):
        """Court (blue) + surround (green), filled in. Excludes the crowd."""
        surface = cv2.bitwise_or(cv2.inRange(hsv, (45, 40, 80), (70, 140, 200)),
                                 cv2.inRange(hsv, (98, 35, 90), (118, 140, 255)))
        surface = cv2.morphologyEx(surface, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        roi = np.zeros_like(surface)
        contours, _ = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(roi, [cv2.convexHull(max(contours, key=cv2.contourArea))],
                             -1, 255, -1)
        return cv2.erode(roi, np.ones((31, 31), np.uint8))

    def motion(self, frame):
        """Double frame difference: pixels that differ from BOTH previous frames.

        Unlike background subtraction this ignores the 1-px shimmer of court
        lines when the camera pans slightly, and marks a moving object only at
        its current position (not where it was).
        """
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(self.history) < 2:
            self.history.append(grey)
            return None
        d1 = cv2.absdiff(grey, self.history[-1])
        d2 = cv2.absdiff(grey, self.history[-2])
        self.history.append(grey)
        moving = cv2.inRange(cv2.min(d1, d2), 25, 255)
        return cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    def _candidates(self, frame):
        fg = self.motion(frame)
        if fg is None:
            return [], np.zeros(frame.shape[:2], np.uint8)
        if cv2.countNonZero(fg) > self.SCENE_CUT * fg.size:
            self.cand_hist.clear()
            self.pos = None
            return [], fg

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = self.play_area(hsv)
        if cv2.countNonZero(roi) < self.MIN_ROI * roi.size:
            return [], roi

        tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, self.TOPHAT)
        bright = cv2.bitwise_and(cv2.inRange(tophat, 30, 255),
                                 cv2.inRange(hsv[..., 1], 0, 150))
        lo, hi = bb.HSV_RANGES["ball"]
        yellow = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.bitwise_and(fg, cv2.bitwise_or(bright, yellow))
        mask = cv2.bitwise_and(mask, roi)
        mask = cv2.dilate(mask, None, iterations=1)

        merged = cv2.dilate(fg, self.MERGE)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(merged)
        players = np.isin(labels, [k for k in range(1, n)
                                   if stats[k, cv2.CC_STAT_AREA] > self.PLAYER_AREA])
        players = cv2.dilate(players.astype(np.uint8), self.MERGE)

        cands = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if not (self.MIN_AREA <= max(area, w * h / 2) <= self.MAX_AREA):
                continue
            if max(w, h) / max(min(w, h), 1) > self.MAX_ASPECT:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if players[int(cy), int(cx)]:
                continue
            cands.append(((int(cx), int(cy)), max(r, 1.5)))
        return cands, mask

    def _start_track(self, cands):
        """Three detections in a row moving at a near-constant velocity."""
        if len(self.cand_hist) < 2:
            return None
        older, prev = self.cand_hist[-2], self.cand_hist[-1]
        best, best_err = None, None
        for (c2, r) in cands:
            for (c1, _) in prev:
                v2 = (c2[0] - c1[0], c2[1] - c1[1])
                step = math.hypot(*v2)
                if not self.MIN_STEP <= step <= self.MAX_STEP:
                    continue
                for (c0, _) in older:
                    v1 = (c1[0] - c0[0], c1[1] - c0[1])
                    err = math.dist(v1, v2)
                    if err <= max(6, 0.3 * step) and (best is None or err < best_err):
                        best, best_err = ((c2, r), v2), err
        return best

    def detect(self, frame):
        cands, mask = self._candidates(frame)

        best = None
        if self.pos is not None:
            k = self.missed + 1
            pred = (self.pos[0] + self.vel[0] * k, self.pos[1] + self.vel[1] * k)
            gate = self.GATE + self.GATE_GROWTH * self.missed
            near = [c for c in cands if math.dist(c[0], pred) <= gate]
            if near:
                best = min(near, key=lambda c: math.dist(c[0], pred))
                self.vel = ((best[0][0] - self.pos[0]) / k, (best[0][1] - self.pos[1]) / k)
            else:
                self.missed += 1
                if self.missed > self.MAX_COAST:
                    self.pos = None
        if self.pos is None:
            started = self._start_track(cands)
            if started:
                best, self.vel = started
        self.cand_hist.append(cands)

        if best is None:
            return None, mask
        self.pos, self.missed = best[0], 0
        centre, r = best
        return {"centre": centre, "radius": r, "area": math.pi * r * r}, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", help="default: v1/output/<name>_tracked_<mode>.mp4")
    ap.add_argument("--mode", choices=["broadcast", "original"], default="broadcast")
    ap.add_argument("--show", action="store_true", help="also show a live preview")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    name = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or os.path.join(OUTPUT_DIR, f"{name}_tracked_{args.mode}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = BroadcastDetector() if args.mode == "broadcast" else OriginalDetector()
    trail = deque(maxlen=TRAIL_LENGTH)
    frames = hits = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        ball, mask = detector.detect(frame)

        out = frame.copy()
        if ball:
            hits += 1
            trail.appendleft(ball["centre"])
            bb.draw_ball(out, ball)
            cv2.circle(out, ball["centre"], int(ball["radius"]) + 10, bb.BALL, 1)
        else:
            trail.appendleft(None)
        for i in range(1, len(trail)):
            if trail[i - 1] and trail[i]:
                thickness = int(math.sqrt(TRAIL_LENGTH / float(i + 1)) * 1.5) + 1
                cv2.line(out, trail[i - 1], trail[i], TRAIL, thickness)

        bb.put_text(out, f"mode: {args.mode} | frame {frames}/{total} | "
                         f"ball: {'yes' if ball else 'no'} | detected in "
                         f"{100 * hits / frames:.0f}% of frames",
                    (10, 22), bb.WHITE, 0.55)
        writer.write(out)

        if args.show:
            cv2.imshow("Ball tracking", out)
            cv2.imshow("Mask", mask)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        elif frames % 300 == 0:
            print(f"  {frames}/{total} frames")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"{args.mode}: ball found in {hits}/{frames} frames "
          f"({100 * hits / max(frames, 1):.1f}%), {frames / (time.time() - start):.0f} fps")
    print("saved", out_path)


if __name__ == "__main__":
    main()
