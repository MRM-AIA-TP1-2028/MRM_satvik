"""Real-time ArUco marker detection and pose estimation.

Detects 4x4, 5x5 and 6x6 ArUco markers from the webcam, estimates each
marker's pose (rotation + translation) and depth, and draws it on the video.

Usage:
    conda activate SANDBOX-MAX
    python aruco_elaborate.py --marker-size 0.05
    python aruco_elaborate.py --marker-size 0.05 --size 6x6=0.08   # per-dictionary size
    python aruco_elaborate.py --source some_video.mp4              # file instead of webcam

Keys:  q / Esc = quit,  s = save screenshot
"""

import argparse
import os
import time

import cv2
import numpy as np

# BGR colours used for each dictionary's outline and label.
DICT_COLOURS = {
    "4x4": (0, 200, 255),   # orange
    "5x5": (255, 120, 0),   # blue
    "6x6": (60, 220, 60),   # green
}


# --------------------------------------------------------------------------
# Camera calibration
# --------------------------------------------------------------------------
def approximate_camera_matrix(width, height):
    """Rough intrinsics when no calibration exists: f ~ image width, centre = image centre."""
    f = float(width)
    K = np.array([[f, 0, width / 2.0],
                  [0, f, height / 2.0],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


def load_calibration(path, width, height):
    """Load K and distortion from calibrate.py output, or fall back to an approximation.

    Returns (K, dist, calibrated: bool, note: str).
    """
    if path and os.path.isfile(path):
        data = np.load(path)
        K = data["K"].astype(np.float64)
        dist = data["dist"].astype(np.float64)
        note = f"calibration: {path}"
        if "image_size" in data:
            cw, ch = (int(v) for v in data["image_size"])
            if (cw, ch) != (width, height):
                # Intrinsics are in pixels; rescale if the resolution differs
                # but the aspect ratio is the same (same sensor crop).
                sx, sy = width / cw, height / ch
                if abs(sx - sy) < 0.01:
                    K = K.copy()
                    K[0, :] *= sx
                    K[1, :] *= sy
                    note += f" (rescaled from {cw}x{ch})"
                else:
                    note += f" (WARNING: calibrated at {cw}x{ch}, running at {width}x{height})"
        return K, dist, True, note

    K, dist = approximate_camera_matrix(width, height)
    return K, dist, False, "NO calibration file - depth is approximate (run calibrate.py)"


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def make_detectors(dict_size):
    """One ArucoDetector each for 4x4, 5x5 and 6x6.

    dict_size (50, 100, 250 or 1000) is how many markers each dictionary holds.
    Smaller dictionaries are the first entries of larger ones, so 250 also
    detects every ID from the 50 and 100 versions.
    """
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    aruco = cv2.aruco
    dictionaries = {
        50:   (aruco.DICT_4X4_50,   aruco.DICT_5X5_50,   aruco.DICT_6X6_50),
        100:  (aruco.DICT_4X4_100,  aruco.DICT_5X5_100,  aruco.DICT_6X6_100),
        250:  (aruco.DICT_4X4_250,  aruco.DICT_5X5_250,  aruco.DICT_6X6_250),
        1000: (aruco.DICT_4X4_1000, aruco.DICT_5X5_1000, aruco.DICT_6X6_1000),
    }
    d4, d5, d6 = dictionaries[dict_size]

    return {
        "4x4": aruco.ArucoDetector(aruco.getPredefinedDictionary(d4), params),
        "5x5": aruco.ArucoDetector(aruco.getPredefinedDictionary(d5), params),
        "6x6": aruco.ArucoDetector(aruco.getPredefinedDictionary(d6), params),
    }


def detect_all(gray, detectors):
    """Run every detector and return all detections in one list.

    Each detection is a dict: {"dict": "5x5", "id": 12, "corners": (4, 2) float32}.
    Corner order is top-left, top-right, bottom-right, bottom-left (on the marker).
    """
    found = []
    for name, detector in detectors.items():
        corners, ids, _rejected = detector.detectMarkers(gray)
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            found.append({"dict": name, "id": int(i), "corners": c.reshape(4, 2)})
    return found


# --------------------------------------------------------------------------
# Pose
# --------------------------------------------------------------------------
def marker_object_points(size):
    """3D corners of a marker of side `size`, centred at the origin, z = 0.

    Marker frame: X right, Y up, Z out of the printed face. Order matches the
    detector: top-left, top-right, bottom-right, bottom-left.
    """
    h = size / 2.0
    return np.array([[-h, h, 0],
                     [h, h, 0],
                     [h, -h, 0],
                     [-h, -h, 0]], dtype=np.float32)


def estimate_pose(corners, size, K, dist):
    """Return (rvec, tvec) for one marker, or None if solvePnP fails."""
    ok, rvec, tvec = cv2.solvePnP(
        marker_object_points(size),
        corners.astype(np.float32),
        K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None
    return rvec, tvec


def rotation_to_euler(R):
    """Rotation matrix -> (roll, pitch, yaw) in degrees, Z-Y-X convention."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: pitch = +-90 deg
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def process_frame(frame, detectors, sizes, default_size, K, dist):
    """Detect markers and estimate pose. Returns a list of result dicts."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    results = []
    for det in detect_all(gray, detectors):
        size = sizes.get(det["dict"], default_size)
        pose = estimate_pose(det["corners"], size, K, dist)
        if pose is None:
            continue
        rvec, tvec = pose
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()
        det.update({
            "size": size,
            "rvec": rvec,
            "tvec": tvec,
            "position": t,                     # metres, camera frame
            "depth": float(t[2]),              # along the optical axis
            "distance": float(np.linalg.norm(t)),
            "euler": rotation_to_euler(R),     # roll, pitch, yaw (deg)
        })
        results.append(det)
    return results


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
LABEL_FONT, LABEL_SCALE, LABEL_THICK = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1


def label_size(lines):
    """(width, line_height, total_height) of a text block in pixels."""
    sizes = [cv2.getTextSize(s, LABEL_FONT, LABEL_SCALE, LABEL_THICK)[0] for s in lines]
    w = max(s[0] for s in sizes) + 8
    lh = max(s[1] for s in sizes) + 6
    return w, lh, lh * len(lines) + 4


def put_label(img, lines, x, y, colour):
    """Draw several lines of text on a semi-transparent dark box at (x, y)."""
    w, lh, h = label_size(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x + 4, y + lh * (i + 1)), LABEL_FONT, LABEL_SCALE,
                    colour if i == 0 else (255, 255, 255), LABEL_THICK, cv2.LINE_AA)


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_label(box, w, h, img_w, img_h, occupied, gap=8):
    """Pick a spot for a w x h label next to a marker's bounding box.

    Tries right, left, below, above; returns the first that stays on screen and
    does not overlap any marker or label already in `occupied`. Falls back to
    the right-hand spot clamped onto the screen.
    """
    x0, y0, x1, y1 = box
    candidates = [
        (x1 + gap, y0),          # right
        (x0 - gap - w, y0),      # left
        (x0, y1 + gap),          # below
        (x0, y0 - gap - h),      # above
    ]
    for x, y in candidates:
        rect = (x, y, x + w, y + h)
        on_screen = x >= 0 and y >= 0 and x + w <= img_w and y + h <= img_h
        if on_screen and not any(_overlaps(rect, o) for o in occupied):
            return int(x), int(y)
    x, y = candidates[0]
    return int(min(max(x, 0), img_w - w)), int(min(max(y, 0), img_h - h))


def draw_results(frame, results, K, dist):
    img_h, img_w = frame.shape[:2]
    boxes = []
    for r in results:
        colour = DICT_COLOURS.get(r["dict"], (255, 255, 255))
        pts = r["corners"].astype(np.int32)
        boxes.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
        cv2.polylines(frame, [pts], True, colour, 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(int(v) for v in pts[0]), 4, (0, 0, 255), -1)  # top-left corner
        cv2.drawFrameAxes(frame, K, dist, r["rvec"], r["tvec"], r["size"] * 0.6, 2)

    occupied = list(boxes)  # labels must not cover any marker
    for r, box in zip(results, boxes):
        colour = DICT_COLOURS.get(r["dict"], (255, 255, 255))
        x, y, z = r["position"]
        roll, pitch, yaw = r["euler"]
        lines = [
            f"{r['dict']} id {r['id']}",
            f"depth {r['depth'] * 100:.1f} cm  dist {r['distance'] * 100:.1f} cm",
            f"xyz ({x * 100:.1f}, {y * 100:.1f}, {z * 100:.1f}) cm",
            f"R {roll:.0f}  P {pitch:.0f}  Y {yaw:.0f} deg",
        ]
        w, _, h = label_size(lines)
        lx, ly = place_label(box, w, h, img_w, img_h, occupied)
        occupied.append((lx, ly, lx + w, ly + h))
        put_label(frame, lines, lx, ly, colour)


def draw_hud(frame, fps, calibrated, n_markers):
    status = "calibrated" if calibrated else "UNCALIBRATED (approx. depth)"
    colour = (0, 255, 0) if calibrated else (0, 0, 255)
    cv2.putText(frame, f"FPS {fps:.1f}   markers {n_markers}   {status}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    cv2.putText(frame, "q: quit   s: screenshot", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def parse_sizes(items):
    """['6x6=0.08', '4x4=0.03'] -> {'6x6': 0.08, '4x4': 0.03}"""
    sizes = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if name not in ("4x4", "5x5", "6x6") or not value:
            raise SystemExit(f"Bad --size '{item}'. Use e.g. --size 6x6=0.08")
        sizes[name] = float(value)
    return sizes


def open_source(source, width, height):
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {source!r}. "
                         "On macOS, allow camera access for your terminal / VS Code.")
    return cap


def main():
    ap = argparse.ArgumentParser(description="ArUco pose estimation (4x4, 5x5, 6x6)")
    ap.add_argument("--source", default="0", help="camera index or video file (default 0)")
    ap.add_argument("--marker-size", type=float, default=0.05,
                    help="printed marker side (black square) in metres (default 0.05)")
    ap.add_argument("--size", action="append", metavar="DICT=METRES",
                    help="per-dictionary marker size, e.g. --size 6x6=0.08 (repeatable)")
    ap.add_argument("--calib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "calibration.npz"),
                    help="calibration file from calibrate.py (default: next to this script)")
    ap.add_argument("--dict-size", type=int, default=250, choices=[50, 100, 250, 1000],
                    help="markers per dictionary (default 250)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    sizes = parse_sizes(args.size)
    cap = open_source(args.source, args.width, args.height)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read a frame from the video source.")
    h, w = frame.shape[:2]

    K, dist, calibrated, note = load_calibration(args.calib, w, h)
    detectors = make_detectors(args.dict_size)

    print(f"Resolution {w}x{h}, dictionaries 4x4, 5x5, 6x6 (_{args.dict_size})")
    print(note)
    print("Marker size:", args.marker_size, "m", sizes if sizes else "")

    fps, last = 0.0, time.perf_counter()
    window = "ArUco pose"
    while ok:
        results = process_frame(frame, detectors, sizes, args.marker_size, K, dist)
        draw_results(frame, results, K, dist)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
        last = now
        draw_hud(frame, fps, calibrated, len(results))

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            name = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, frame)
            print("saved", name)

        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
