"""ArUco pose estimation: 4x4, 5x5 and 6x6 markers from the webcam.

Same pose features as aruco_elaborate.py (calibration with rescaling, sub-pixel
corners, roll/pitch/yaw, depth and distance) without the display extras.

Run:   python aruco_medium.py        (q = quit)
"""

import os

import cv2
import numpy as np

MARKER_SIZE = 0.05           # printed black-square side in metres
SIZES = {}                   # per-family override, e.g. {"6x6": 0.08}
CALIB_FILE = "calibration.npz"
WIDTH, HEIGHT = 640, 480


def load_calibration(path, width, height):
    """Return (K, dist, calibrated). Falls back to a rough guess with no file."""
    if not os.path.isfile(path):
        K = np.array([[width, 0, width / 2],
                      [0, width, height / 2],
                      [0, 0, 1]], dtype=np.float64)
        return K, np.zeros(5), False

    data = np.load(path)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)
    cw, ch = (int(v) for v in data["image_size"])
    sx, sy = width / cw, height / ch
    if abs(sx - sy) < 0.01:                  # same aspect ratio: rescale K
        K[0, :] *= sx
        K[1, :] *= sy
    else:
        print(f"WARNING: calibrated at {cw}x{ch}, running at {width}x{height}")
    return K, dist, True


def make_detectors():
    """One detector per family. The _1000 versions detect marker IDs 0-999."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    dict4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    dict5 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    dict6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    return {
        "4x4": cv2.aruco.ArucoDetector(dict4, params),
        "5x5": cv2.aruco.ArucoDetector(dict5, params),
        "6x6": cv2.aruco.ArucoDetector(dict6, params),
    }


def marker_points(size):
    """3D corners of the marker, centred at the origin, in detector order."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


def rotation_to_euler(rvec):
    """Rotation vector -> (roll, pitch, yaw) in degrees."""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                                    # gimbal lock: pitch = +-90 deg
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the webcam. Check camera permissions.")
    h, w = frame.shape[:2]

    K, dist, calibrated = load_calibration(CALIB_FILE, w, h)
    if not calibrated:
        print("No calibration file: depth is approximate (run calibrate.py)")
    detectors = make_detectors()

    while ok:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for name, detector in detectors.items():
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            size = SIZES.get(name, MARKER_SIZE)
            for c, marker_id in zip(corners, ids.flatten()):
                found, rvec, tvec = cv2.solvePnP(marker_points(size), c[0], K, dist,
                                                 flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not found:
                    continue
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, size / 2)

                x, y, z = tvec.flatten() * 100                   # cm
                distance = np.linalg.norm(tvec) * 100            # cm
                roll, pitch, yaw = rotation_to_euler(rvec)

                px, py = c[0][0].astype(int)
                lines = [
                    f"{name} id {marker_id}",
                    f"depth {z:.1f} cm  dist {distance:.1f} cm",
                    f"xyz ({x:.1f}, {y:.1f}, {z:.1f}) cm",
                    f"R {roll:.0f}  P {pitch:.0f}  Y {yaw:.0f} deg",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(frame, text, (px, py - 60 + 18 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if not calibrated:
            cv2.putText(frame, "UNCALIBRATED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("aruco medium", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
