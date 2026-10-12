import cv2
import numpy as np

MARKER_SIZE = 0.05  # side of the black square in metres (measure your printout)

# One detector per dictionary: 4x4, 5x5, 6x6
detectors = {
    name: cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(d))
    for name, d in [("4x4", cv2.aruco.DICT_4X4_250),
                    ("5x5", cv2.aruco.DICT_5X5_250),
                    ("6x6", cv2.aruco.DICT_6X6_250)]
}

# Marker corners in 3D (marker centre = origin), same order as detectMarkers returns
s = MARKER_SIZE / 2
obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ok, frame = cap.read()
h, w = frame.shape[:2]

# Camera intrinsics: use calibration.npz if it exists, else a rough guess
try:
    calib = np.load("calibration.npz")
    K, dist = calib["K"], calib["dist"]
except FileNotFoundError:
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

while ok:
    for name, detector in detectors.items():
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            continue
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for c, marker_id in zip(corners, ids.flatten()):
            _, rvec, tvec = cv2.solvePnP(obj_pts, c[0], K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_SIZE / 2)

            x, y, z = tvec.flatten() * 100  # cm
            rx, ry, rz = np.degrees(rvec.flatten())
            px, py = c[0][0].astype(int)
            cv2.putText(frame, f"{name} id {marker_id}  depth {z:.1f}cm", (px, py - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"pos ({x:.1f},{y:.1f},{z:.1f})cm  rot ({rx:.0f},{ry:.0f},{rz:.0f})deg",
                        (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("aruco pose", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    ok, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
