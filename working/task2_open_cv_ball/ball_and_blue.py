"""Blue object selection/tracking and tennis ball detection from a live camera.

Blue objects (any shape) and the tennis ball are found with separate HSV
colour masks and treated as separate objects:

  Blue object  -> area (px^2), bounding box, trail of its centre, and its
                  position relative to the tennis ball (if one is visible).
  Tennis ball  -> centre, radius (px) and area (px^2).

Usage:
    conda activate SANDBOX-MAX
    python ball_and_blue.py                 # default webcam
    python ball_and_blue.py --source 1      # another camera
    python ball_and_blue.py --source vid.mp4

Mouse:  left-click a blue object = select it (only that one is tracked)
        right-click              = clear selection (track all blue objects)
Keys:   t = HSV tuning sliders,  m = show masks,  c = clear trails,
        s = screenshot,          q / Esc = quit
"""

import argparse
import math
import time
from collections import deque

import cv2
import numpy as np

WINDOW = "Blue object + tennis ball"
TUNE_WINDOW = "HSV tuning"

# HSV ranges (OpenCV: H 0-179, S/V 0-255). Adjust live with the 't' sliders.
HSV_RANGES = {
    "blue": [[100, 120, 50], [130, 255, 255]],
    "ball": [[25, 70, 80], [45, 255, 255]],     # optic yellow / green
}

MIN_BLUE_AREA = 800          # px^2, ignore smaller blue blobs (noise)
MIN_BALL_RADIUS = 8          # px
MIN_CIRCULARITY = 0.65       # contour area / enclosing-circle area for the ball
TRAIL_LENGTH = 64            # number of past centres kept per blue object
MAX_MATCH_DIST = 90          # px, max centre jump between frames for the same ID
MAX_MISSED = 15              # frames an object can disappear before its ID is dropped

BLUE = (255, 150, 40)
BALL = (0, 255, 255)
SELECTED = (0, 0, 255)
LINK = (255, 0, 255)
WHITE = (255, 255, 255)

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# --------------------------------------------------------------------------
# Masks and detection
# --------------------------------------------------------------------------
def colour_mask(hsv, name):
    lo, hi = HSV_RANGES[name]
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    return mask


def detect_blue(mask):
    """Every blue blob above MIN_BLUE_AREA. No shape is assumed."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_BLUE_AREA:
            continue
        m = cv2.moments(c)
        centre = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        objects.append({"contour": c, "area": area,
                        "bbox": cv2.boundingRect(c), "centre": centre})
    return objects


def detect_ball(mask):
    """Largest sufficiently round yellow-green blob, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < MIN_BALL_RADIUS:
            continue
        circularity = cv2.contourArea(c) / (math.pi * r * r)
        if circularity < MIN_CIRCULARITY:
            continue
        if best is None or r > best["radius"]:
            best = {"centre": (int(x), int(y)), "radius": r}
    if best:
        best["area"] = math.pi * best["radius"] ** 2
    return best


# --------------------------------------------------------------------------
# Tracking: nearest-centre matching gives each blue object a stable ID
# --------------------------------------------------------------------------
class CentroidTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}   # id -> {"centre", "missed", "trail", "obj"}

    def update(self, objects):
        unmatched = list(range(len(objects)))
        # Greedily pair existing tracks with the closest new detection.
        pairs = sorted(
            (math.dist(t["centre"], objects[i]["centre"]), tid, i)
            for tid, t in self.tracks.items() for i in unmatched)
        used_tracks = set()
        for d, tid, i in pairs:
            if d > MAX_MATCH_DIST or tid in used_tracks or i not in unmatched:
                continue
            self._assign(tid, objects[i])
            used_tracks.add(tid)
            unmatched.remove(i)

        for tid in list(self.tracks):
            if tid not in used_tracks:
                self.tracks[tid]["missed"] += 1
                self.tracks[tid]["obj"] = None
                if self.tracks[tid]["missed"] > MAX_MISSED:
                    del self.tracks[tid]

        for i in unmatched:
            self.tracks[self.next_id] = {"trail": deque(maxlen=TRAIL_LENGTH)}
            self._assign(self.next_id, objects[i])
            self.next_id += 1

    def _assign(self, tid, obj):
        t = self.tracks[tid]
        t.update(centre=obj["centre"], missed=0, obj=obj)
        t["trail"].appendleft(obj["centre"])

    def visible(self):
        return {tid: t for tid, t in self.tracks.items() if t["obj"] is not None}

    def clear_trails(self):
        for t in self.tracks.values():
            t["trail"].clear()


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def put_text(img, text, org, colour=WHITE, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def relative_position(blue_c, ball_c):
    """Blue object position w.r.t. the ball (image y grows downward)."""
    dx, dy = blue_c[0] - ball_c[0], blue_c[1] - ball_c[1]
    horiz = "right" if dx > 0 else "left"
    vert = "below" if dy > 0 else "above"
    angle = math.degrees(math.atan2(-dy, dx))   # 0 deg = right, 90 deg = up
    return dx, dy, math.hypot(dx, dy), angle, f"{abs(dx)}px {horiz}, {abs(dy)}px {vert}"


def draw_trail(img, trail, colour):
    for i in range(1, len(trail)):
        thickness = int(math.sqrt(TRAIL_LENGTH / float(i + 1)) * 2) + 1
        cv2.line(img, trail[i - 1], trail[i], colour, thickness)


def draw_blue(img, tid, track, ball, selected):
    obj = track["obj"]
    colour = SELECTED if selected else BLUE
    x, y, w, h = obj["bbox"]
    cx, cy = obj["centre"]

    cv2.drawContours(img, [obj["contour"]], -1, colour, 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
    cv2.circle(img, (cx, cy), 4, colour, -1)
    draw_trail(img, track["trail"], colour)

    label = f"Blue #{tid}{' [SEL]' if selected else ''}"
    put_text(img, label, (x, max(y - 26, 12)), colour)
    put_text(img, f"Area: {obj['area']:.0f} px^2  Box: {w}x{h}", (x, max(y - 8, 28)), colour)

    lines = []
    if ball:
        dx, dy, dist, angle, desc = relative_position(obj["centre"], ball["centre"])
        cv2.line(img, ball["centre"], (cx, cy), LINK, 2)
        mid = ((cx + ball["centre"][0]) // 2, (cy + ball["centre"][1]) // 2)
        put_text(img, f"{dist:.0f}px", mid, LINK)
        lines = [f"wrt ball: dx={dx:+d} dy={dy:+d}",
                 f"{desc}", f"dist={dist:.0f}px angle={angle:.0f}deg"]
    else:
        lines = ["wrt ball: no ball"]
    for i, text in enumerate(lines):
        put_text(img, text, (x, y + h + 16 + 16 * i), colour, 0.45)


def draw_ball(img, ball):
    c, r = ball["centre"], int(ball["radius"])
    cv2.circle(img, c, r, BALL, 2)
    cv2.circle(img, c, 4, BALL, -1)
    put_text(img, "Tennis ball", (c[0] - r, c[1] - r - 42), BALL)
    put_text(img, f"Centre: ({c[0]}, {c[1]})", (c[0] - r, c[1] - r - 26), BALL)
    put_text(img, f"R: {ball['radius']:.1f}px  Area: {ball['area']:.0f} px^2",
             (c[0] - r, c[1] - r - 10), BALL)


# --------------------------------------------------------------------------
# HSV tuning sliders
# --------------------------------------------------------------------------
def open_tuner():
    cv2.namedWindow(TUNE_WINDOW)
    for name, (lo, hi) in HSV_RANGES.items():
        for ch, lo_v, hi_v, top in zip("HSV", lo, hi, (179, 255, 255)):
            cv2.createTrackbar(f"{name} {ch} min", TUNE_WINDOW, lo_v, top, lambda v: None)
            cv2.createTrackbar(f"{name} {ch} max", TUNE_WINDOW, hi_v, top, lambda v: None)


def read_tuner():
    for name, (lo, hi) in HSV_RANGES.items():
        for k, ch in enumerate("HSV"):
            lo[k] = cv2.getTrackbarPos(f"{name} {ch} min", TUNE_WINDOW)
            hi[k] = cv2.getTrackbarPos(f"{name} {ch} max", TUNE_WINDOW)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {args.source!r}")

    tracker = CentroidTracker()
    state = {"selected": None, "click": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["selected"] = None

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    tuning, show_masks = False, False
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if isinstance(src, int):
            frame = cv2.flip(frame, 1)          # mirror the webcam view

        if tuning:
            read_tuner()
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        blue_mask = colour_mask(hsv, "blue")
        ball_mask = colour_mask(hsv, "ball")

        ball = detect_ball(ball_mask)
        tracker.update(detect_blue(blue_mask))
        visible = tracker.visible()

        # A click selects the blue object whose contour contains the point.
        if state["click"]:
            pt = tuple(float(v) for v in state["click"])
            state["selected"] = next(
                (tid for tid, t in visible.items()
                 if cv2.pointPolygonTest(t["obj"]["contour"], pt, False) >= 0), None)
            state["click"] = None
        if state["selected"] is not None and state["selected"] not in tracker.tracks:
            state["selected"] = None            # selected object has been lost

        out = frame.copy()
        for tid, track in visible.items():
            if state["selected"] is None or tid == state["selected"]:
                draw_blue(out, tid, track, ball, tid == state["selected"])
        if ball:
            draw_ball(out, ball)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        mode = (f"selected #{state['selected']}" if state["selected"] is not None
                else f"tracking all ({len(visible)} blue)")
        put_text(out, f"FPS {fps:.0f} | {mode} | ball: {'yes' if ball else 'no'}",
                 (10, 20), WHITE, 0.55)
        put_text(out, "L-click: select  R-click: all  t: tune  m: masks  c: clear  q: quit",
                 (10, out.shape[0] - 10), WHITE, 0.45)

        cv2.imshow(WINDOW, out)
        if show_masks:
            cv2.imshow("Masks (blue | ball)", np.hstack([blue_mask, ball_mask]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("t"):
            tuning = not tuning
            open_tuner() if tuning else cv2.destroyWindow(TUNE_WINDOW)
        elif key == ord("m"):
            show_masks = not show_masks
            if not show_masks:
                cv2.destroyWindow("Masks (blue | ball)")
        elif key == ord("c"):
            tracker.clear_trails()
        elif key == ord("s"):
            name = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(name, out)
            print("saved", name)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
