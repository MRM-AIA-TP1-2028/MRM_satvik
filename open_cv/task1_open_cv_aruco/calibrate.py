"""Camera calibration with a ChArUco board -> calibration.npz.

A ChArUco board is a chessboard with an ArUco marker in every white square.
The markers tell OpenCV which corner is which, so the board can be partly out
of view or seen at an angle and every visible corner is still usable.

Steps:
    1. python calibrate.py --make-board      # writes charuco_board.png
       Print it at 100% scale (no "fit to page"), tape it flat to something
       stiff, and measure one chessboard square and one marker in metres.
    2. python calibrate.py --square 0.035 --marker 0.026
       Hold the board in front of the camera. Press SPACE to capture a view
       when the corners are drawn. Capture 15-25 views: near and far, tilted
       left/right/up/down, and with the board in every part of the image
       (the dots show where you have already covered).
       Press c to calibrate and save, u to toggle an undistorted preview to
       check the result, q to quit.
    3. Run any of the aruco_*.py scripts - they load calibration.npz from this
       folder automatically.

Recalibrate from the saved views without the camera:
    python calibrate.py --from-images calib_images

Output: calibration.npz with
    K           3x3 camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] in px
    dist        distortion coefficients [k1, k2, p1, p2, k3]
    image_size  (width, height) the calibration was made at
    rms         reprojection error in px (below ~0.5 is good)

The square/marker sizes only set the scale of the board; the camera matrix
and distortion do not depend on them, but measure them anyway so the saved
board poses are in real metres.
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(HERE, "calibration.npz")
DEFAULT_IMAGE_DIR = os.path.join(HERE, "calib_images")

BOARD_SQUARES = (7, 5)               # squares across, down
DICTIONARY = cv2.aruco.DICT_5X5_100
MIN_CORNERS = 8                      # corners needed for a view to count
MIN_VIEWS = 10                       # views needed before calibrating


def make_board(square, marker):
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, square, marker, dictionary)


def save_board_image(board, path):
    """Board image for an A4 page in landscape at ~300 dpi."""
    img = board.generateImage((3300, 2300), marginSize=150)
    cv2.imwrite(path, img)
    print(f"saved {path} - print at 100% scale and measure a square and a marker")


def detect(detector, board, gray):
    """Return (object_points, image_points, corners, ids) or None if too few corners."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) < MIN_CORNERS:
        return None
    return obj, img, corners, ids


def calibrate(views, image_size, output, free_k3=False):
    """Calibrate from [(object_points, image_points)] and save calibration.npz.

    k3 is fixed at 0 unless free_k3: for normal webcam lenses it only fits
    noise and can bend the image corners by several pixels.
    """
    obj_pts = [v[0] for v in views]
    img_pts = [v[1] for v in views]
    flags = 0 if free_k3 else cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None,
                                                     flags=flags)

    # Per-view error helps spot a bad (blurred) capture.
    errors = []
    for obj, img, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=2)))))

    np.savez(output, K=K, dist=dist.ravel(), image_size=np.array(image_size), rms=rms)

    k1, k2, p1, p2, k3 = dist.ravel()[:5]
    print(f"\ncalibrated from {len(views)} views at {image_size[0]}x{image_size[1]}")
    print(f"  reprojection error: {rms:.3f} px  ({'good' if rms < 0.5 else 'ok' if rms < 1.0 else 'poor - recapture'})")
    print(f"  fx={K[0, 0]:.1f}  fy={K[1, 1]:.1f}  cx={K[0, 2]:.1f}  cy={K[1, 2]:.1f}")
    print(f"  k1={k1:+.4f}  k2={k2:+.4f}  p1={p1:+.4f}  p2={p2:+.4f}  k3={k3:+.4f}")
    worst = int(np.argmax(errors))
    print(f"  worst view: #{worst + 1} ({errors[worst]:.2f} px)")
    print(f"saved {output}")
    return K, dist, rms


def from_images(folder, board, output, free_k3=False):
    detector = cv2.aruco.CharucoDetector(board)
    paths = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
    if not paths:
        raise SystemExit(f"no .png/.jpg images in {folder}")
    views, size = [], None
    for p in paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != size:
            print(f"  skipped {os.path.basename(p)}: different resolution")
            continue
        found = detect(detector, board, gray)
        if found:
            views.append(found[:2])
        else:
            print(f"  skipped {os.path.basename(p)}: board not found")
    if len(views) < MIN_VIEWS:
        raise SystemExit(f"only {len(views)} usable views, need at least {MIN_VIEWS}")
    calibrate(views, size, output, free_k3)


def live(args, board):
    detector = cv2.aruco.CharucoDetector(board)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera. On macOS, allow camera access "
                         "for your terminal / VS Code.")
    h, w = frame.shape[:2]
    os.makedirs(args.image_dir, exist_ok=True)
    print(f"camera at {w}x{h}. SPACE = capture, c = calibrate + save, "
          f"u = undistorted preview, q = quit")

    views, covered = [], []
    K = dist = None
    undistort = False
    flash = 0
    while ok:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = detect(detector, board, gray)

        shown = cv2.undistort(frame, K, dist) if (undistort and K is not None) else frame.copy()
        if not undistort:
            for pt in covered:                       # where earlier views put corners
                cv2.circle(shown, pt, 2, (255, 200, 0), -1)
            if found:
                cv2.aruco.drawDetectedCornersCharuco(shown, found[2], found[3], (0, 255, 0))

        status = (f"views {len(views)}/{MIN_VIEWS}+  corners {len(found[3]) if found else 0}"
                  + ("  [UNDISTORTED]" if undistort else ""))
        cv2.putText(shown, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(shown, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if flash:                                     # brief white border after a capture
            cv2.rectangle(shown, (0, 0), (w - 1, h - 1), (255, 255, 255), 8)
            flash -= 1
        cv2.imshow("calibrate", shown)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            if found:
                views.append(found[:2])
                covered.extend(tuple(int(v) for v in p) for p in found[1].reshape(-1, 2))
                cv2.imwrite(os.path.join(args.image_dir, f"view_{int(time.time() * 1000)}.png"), frame)
                flash = 3
                print(f"  captured view {len(views)} ({len(found[3])} corners)")
            else:
                print(f"  board not found - need at least {MIN_CORNERS} corners")
        if key == ord("c"):
            if len(views) < MIN_VIEWS:
                print(f"  need at least {MIN_VIEWS} views (have {len(views)})")
            else:
                K, dist, _ = calibrate(views, (w, h), args.output, args.free_k3)
                print("  press u to check: straight edges should look straight")
        if key == ord("u"):
            if K is None:
                print("  calibrate first (c)")
            else:
                undistort = not undistort
        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-board", action="store_true", help="write charuco_board.png and exit")
    ap.add_argument("--square", type=float, default=0.035, help="printed square side in metres")
    ap.add_argument("--marker", type=float, default=0.026, help="printed marker side in metres")
    ap.add_argument("--from-images", metavar="FOLDER", help="calibrate from saved images")
    ap.add_argument("--source", default="0", help="camera index or video file (default 0)")
    ap.add_argument("--width", type=int, default=640, help="match the aruco scripts (640)")
    ap.add_argument("--height", type=int, default=480, help="match the aruco scripts (480)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="where captured views are saved")
    ap.add_argument("--free-k3", action="store_true",
                    help="also fit k3 (only for wide-angle / fisheye-like lenses)")
    args = ap.parse_args()

    if args.marker >= args.square:
        raise SystemExit("--marker must be smaller than --square")
    board = make_board(args.square, args.marker)

    if args.make_board:
        save_board_image(board, os.path.join(HERE, "charuco_board.png"))
    elif args.from_images:
        from_images(args.from_images, board, args.output, args.free_k3)
    else:
        live(args, board)


if __name__ == "__main__":
    main()
