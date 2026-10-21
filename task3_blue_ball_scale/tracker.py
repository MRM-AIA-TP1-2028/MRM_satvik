"""Blue object + tennis ball detection and tracking with ArUco scale estimation.

Live camera stream -> detects every blue object (any shape) and a tennis ball,
tracks each blue object over time, and reports sizes in cm using an ArUco
marker of known side length as the scale reference.

Windows
    Camera  - contours, bounding boxes, center trails, ball circle, labels
    Info    - pitch-black window: (x, y) of each blue object, its position
              relative to the ball, distance traversed, distances between
              blue objects
    Masks   - (toggle with 'm') the blue and ball HSV masks

Keys
    q / Esc  quit
    c        clear trails and traversed distances
    r        reset the blue HSV range to the default
    m        toggle mask windows
    click    on a blue object in the Camera window to select its colour
"""

import math
from collections import deque

import cv2
import numpy as np

# Settings: edit these instead of passing command-line options
SOURCE = 0          # 0 = MacBook Air camera (an iPhone via Continuity Camera may be 1)
MARKER_CM = 5.0     # side length of the printed ArUco marker's black square in cm
MIN_BLUE_AREA = 600  # min blue blob area in px^2
MIN_BALL_AREA = 300  # min ball blob area in px^2
FRAME_W, FRAME_H = 1280, 720  # camera can do 1920x1080; 720p fits the screen and runs faster

ARUCO_DICT = cv2.aruco.DICT_4X4_50  # make_marker.py prints from this dictionary

# OpenCV HSV: H in [0, 179], S and V in [0, 255]
DEFAULT_BLUE = (np.array([100, 120, 50]), np.array([130, 255, 255]))
BALL_RANGE = (np.array([25, 70, 70]), np.array([45, 255, 255]))
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
BLUE_OUTLINE = (255, 140, 0)
BOX_COLOR = (0, 255, 255)
BALL_COLOR = (0, 255, 0)
LINK_COLOR = (0, 0, 255)


class ScaleEstimator:
    """Pixels-per-cm from the side length of any visible ArUco marker."""

    def __init__(self, marker_cm, alpha=0.3):
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
        self.marker_cm = marker_cm
        self.alpha = alpha  # smoothing factor for the running estimate
        self.px_per_cm = None
        self.visible = False

    def update(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        corners_all = list(corners) if ids is not None else []

        self.visible = bool(corners_all)
        if corners_all:
            sides = []
            for c in corners_all:
                pts = c.reshape(4, 2)
                sides.extend(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1))
            ppc = float(np.mean(sides)) / self.marker_cm
            if self.px_per_cm is None:
                self.px_per_cm = ppc
            else:
                self.px_per_cm = (1 - self.alpha) * self.px_per_cm + self.alpha * ppc
        return corners_all

    def length(self, px):
        """Format a pixel length in cm when the scale is known, else in px."""
        if self.px_per_cm:
            return f"{px / self.px_per_cm:.1f} cm"
        return f"{px:.0f} px"

    def area(self, area_px):
        if self.px_per_cm:
            return f"{area_px / self.px_per_cm ** 2:.1f} cm^2"
        return f"{area_px:.0f} px^2"


def clean_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)


def centroid(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def find_blue(hsv, lo, hi, min_area):
    """Every blue blob above min_area; no assumption about its shape."""
    mask = clean_mask(cv2.inRange(hsv, lo, hi))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        center = centroid(c)
        if center is None:
            continue
        objects.append({"contour": c, "area_px": area, "center": center})
    return objects, mask


def find_ball(hsv, min_area, min_circularity=0.6, min_fill=0.5):
    """The most ball-like yellow-green blob, or None."""
    mask = clean_mask(cv2.inRange(hsv, *BALL_RANGE))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area < min_area or perimeter == 0:
            continue
        circularity = 4 * math.pi * area / perimeter ** 2
        (x, y), r = cv2.minEnclosingCircle(c)
        fill = area / (math.pi * r * r)
        if circularity < min_circularity or fill < min_fill:
            continue
        score = area * circularity
        if score > best_score:
            best_score = score
            best = {"center": (x, y), "radius": r, "area_px": math.pi * r * r,
                    "circularity": circularity}
    return best, mask


class Track:
    def __init__(self, tid, obj, trail_len):
        self.id = tid
        self.center = obj["center"]
        self.anchor = obj["center"]  # last point counted towards distance
        self.trail = deque([obj["center"]], maxlen=trail_len)
        self.dist_px = 0.0
        self.dist_cm = 0.0
        self.missed = 0
        self.obj = obj


class CentroidTracker:
    """Greedy nearest-centroid matching so each blue object keeps its ID."""

    def __init__(self, max_dist=120, max_missed=15, trail_len=64, min_step=3.0):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.trail_len = trail_len
        self.min_step = min_step  # ignore jitter smaller than this (px)
        self.tracks = []
        self.next_id = 1

    def reset(self):
        self.tracks = []
        self.next_id = 1

    def update(self, objects, px_per_cm):
        pairs = sorted(
            (math.dist(t.center, o["center"]), ti, oi)
            for ti, t in enumerate(self.tracks)
            for oi, o in enumerate(objects)
        )
        used_t, used_o = set(), set()
        for d, ti, oi in pairs:
            if d > self.max_dist or ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            t, o = self.tracks[ti], objects[oi]
            t.center, t.obj, t.missed = o["center"], o, 0
            step = math.dist(t.anchor, t.center)
            if step >= self.min_step:
                t.dist_px += step
                if px_per_cm:
                    t.dist_cm += step / px_per_cm
                t.anchor = t.center
                t.trail.append(t.center)

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
                t.obj = None
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        for oi, o in enumerate(objects):
            if oi not in used_o:
                self.tracks.append(Track(self.next_id, o, self.trail_len))
                self.next_id += 1

        return [t for t in self.tracks if t.obj is not None]


class Pipeline:
    def __init__(self, marker_cm, min_blue_area, min_ball_area):
        self.scale = ScaleEstimator(marker_cm)
        self.tracker = CentroidTracker()
        self.blue_lo, self.blue_hi = (a.copy() for a in DEFAULT_BLUE)
        self.min_blue_area = min_blue_area
        self.min_ball_area = min_ball_area
        self.hsv = None

    def process(self, frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        self.hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners = self.scale.update(gray)
        blue, blue_mask = find_blue(self.hsv, self.blue_lo, self.blue_hi, self.min_blue_area)
        ball, ball_mask = find_ball(self.hsv, self.min_ball_area)
        tracks = self.tracker.update(blue, self.scale.px_per_cm)
        return {"tracks": tracks, "ball": ball, "corners": corners,
                "blue_mask": blue_mask, "ball_mask": ball_mask}

    def select_blue(self, x, y):
        """Centre the blue HSV range on the colour under the clicked pixel."""
        if self.hsv is None:
            return
        h_img, w_img = self.hsv.shape[:2]
        patch = self.hsv[max(y - 3, 0):min(y + 4, h_img), max(x - 3, 0):min(x + 4, w_img)]
        h, s, v = np.median(patch.reshape(-1, 3), axis=0)
        self.blue_lo = np.array([max(h - 10, 0), max(s - 70, 50), max(v - 70, 40)], dtype=np.uint8)
        self.blue_hi = np.array([min(h + 10, 179), 255, 255], dtype=np.uint8)

    def reset_blue(self):
        self.blue_lo, self.blue_hi = (a.copy() for a in DEFAULT_BLUE)


def pt(p):
    return (int(round(p[0])), int(round(p[1])))


def relative(obj_center, ball, scale):
    """Blue object's offset from the ball (+x right, +y down) and distance."""
    dx = obj_center[0] - ball["center"][0]
    dy = obj_center[1] - ball["center"][1]
    d = math.hypot(dx, dy)
    if scale.px_per_cm:
        k = scale.px_per_cm
        return f"dx={dx / k:+.1f} dy={dy / k:+.1f} d={d / k:.1f} cm"
    return f"dx={dx:+.0f} dy={dy:+.0f} d={d:.0f} px"


def put(img, text, org, color=WHITE, size=0.5, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def draw_camera(frame, result, scale):
    tracks, ball = result["tracks"], result["ball"]

    if result["corners"]:
        cv2.aruco.drawDetectedMarkers(frame, result["corners"])

    for t in tracks:
        c = t.obj["contour"]
        cv2.drawContours(frame, [c], -1, BLUE_OUTLINE, 2)
        box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.int32)
        cv2.drawContours(frame, [box], 0, BOX_COLOR, 1)
        if len(t.trail) > 1:
            cv2.polylines(frame, [np.array([pt(p) for p in t.trail], np.int32)], False, BLUE_OUTLINE, 2)
        cx, cy = pt(t.center)
        cv2.circle(frame, (cx, cy), 4, WHITE, -1)
        put(frame, f"B{t.id} {scale.area(t.obj['area_px'])}", (cx + 8, cy - 8))
        if ball:
            cv2.line(frame, (cx, cy), pt(ball["center"]), LINK_COLOR, 1, cv2.LINE_AA)
            put(frame, relative(t.center, ball, scale), (cx + 8, cy + 12), size=0.45)

    if ball:
        bc, r = pt(ball["center"]), int(round(ball["radius"]))
        cv2.circle(frame, bc, r, BALL_COLOR, 2)
        cv2.circle(frame, bc, 4, BALL_COLOR, -1)
        label = f"Ball c=({bc[0]},{bc[1]}) r={r}px A={scale.area(ball['area_px'])}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx = min(max(bc[0] - r, 5), frame.shape[1] - tw - 5)
        put(frame, label, (lx, max(bc[1] - r - 10, 45)), BALL_COLOR)

    if scale.px_per_cm:
        state = "marker" if scale.visible else "held"
        hud = f"Scale: {scale.px_per_cm:.2f} px/cm ({state}, {scale.marker_cm} cm marker)"
    else:
        hud = "Scale: no ArUco marker yet - showing px"
    put(frame, hud, (10, 25), size=0.6)
    put(frame, "click blue to select | c clear | r reset | m masks | q quit",
        (10, frame.shape[0] - 12), GRAY, size=0.45)
    return frame


def draw_info(shape, result, scale, panel_w=380):
    """Pitch-black window: a white map of the scene plus a text panel."""
    h, w = shape[:2]
    canvas = np.zeros((h, w + panel_w, 3), np.uint8)
    tracks, ball = result["tracks"], result["ball"]

    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            cv2.line(canvas, pt(a.center), pt(b.center), GRAY, 1, cv2.LINE_AA)
            mid = ((a.center[0] + b.center[0]) / 2, (a.center[1] + b.center[1]) / 2)
            put(canvas, scale.length(math.dist(a.center, b.center)), pt(mid), GRAY, 0.45)
    if ball:
        cv2.circle(canvas, pt(ball["center"]), int(round(ball["radius"])), WHITE, 2)
        put(canvas, "ball", pt(ball["center"]))
    for t in tracks:
        if len(t.trail) > 1:
            cv2.polylines(canvas, [np.array([pt(p) for p in t.trail], np.int32)], False, GRAY, 1)
        cv2.circle(canvas, pt(t.center), 6, WHITE, -1)
        put(canvas, f"B{t.id}", (pt(t.center)[0] + 9, pt(t.center)[1] - 9))
        if ball:
            cv2.line(canvas, pt(t.center), pt(ball["center"]), WHITE, 1, cv2.LINE_AA)

    x0, y = w + 12, 28
    cv2.line(canvas, (w, 0), (w, h), GRAY, 1)
    put(canvas, f"Blue objects: {len(tracks)}", (x0, y), size=0.6)
    y += 26
    if ball:
        bc = pt(ball["center"])
        put(canvas, f"Ball: ({bc[0]}, {bc[1]}) r={ball['radius']:.0f}px", (x0, y))
        y += 20
        put(canvas, f"      A={scale.area(ball['area_px'])}", (x0, y))
    else:
        put(canvas, "Ball: not detected", (x0, y), GRAY)
    y += 30

    for t in tracks:
        if y > h - 20:
            break
        cx, cy = pt(t.center)
        traversed = f"{t.dist_cm:.1f} cm" if scale.px_per_cm else f"{t.dist_px:.0f} px"
        put(canvas, f"B{t.id}: (x={cx}, y={cy})  A={scale.area(t.obj['area_px'])}", (x0, y))
        y += 20
        if ball:
            put(canvas, f"   wrt ball: {relative(t.center, ball, scale)}", (x0, y), size=0.45)
            y += 20
        put(canvas, f"   traversed: {traversed}", (x0, y), size=0.45)
        y += 26
    return canvas


def main():
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {SOURCE!r}")

    pipe = Pipeline(MARKER_CM, MIN_BLUE_AREA, MIN_BALL_AREA)
    show_masks = False

    cv2.namedWindow("Camera")
    cv2.namedWindow("Info")

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pipe.select_blue(x, y)
            pipe.tracker.reset()

    cv2.setMouseCallback("Camera", on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipe.process(frame)
        cv2.imshow("Camera", draw_camera(frame.copy(), result, pipe.scale))
        cv2.imshow("Info", draw_info(frame.shape, result, pipe.scale))
        if show_masks:
            cv2.imshow("Blue mask", result["blue_mask"])
            cv2.imshow("Ball mask", result["ball_mask"])

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            pipe.tracker.reset()
        elif key == ord("r"):
            pipe.reset_blue()
            pipe.tracker.reset()
        elif key == ord("m"):
            show_masks = not show_masks
            if not show_masks:
                cv2.destroyWindow("Blue mask")
                cv2.destroyWindow("Ball mask")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
