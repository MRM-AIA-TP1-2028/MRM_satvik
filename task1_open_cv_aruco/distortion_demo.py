"""See what lens distortion does to a chessboard.

Five sliders control OpenCV's distortion coefficients [k1, k2, p1, p2, k3].
The window shows what a camera with those values would see when looking
straight at a flat chessboard. Thin red lines are where the squares would be
with a perfect lens (all zeros).

Run:   python distortion_demo.py
Keys:  r = your webcam's calibrated values,  z = all zeros,  q / Esc = quit
"""

import cv2
import numpy as np

W, H = 640, 480
FX = FY = 396.0              # from calibration.npz
CX, CY = W / 2, H / 2
SQUARE = 40                  # chessboard square size in pixels (in the ideal image)

MINE = {"k1": -0.0403, "k2": 0.0049, "p1": -0.0094, "p2": 0.0058, "k3": 0.0224}

# name -> (min, max). Each slider has 200 steps across its range.
RANGES = {
    "k1": (-1.0, 1.0),       # main radial: negative = barrel, positive = pincushion
    "k2": (-1.0, 1.0),       # radial, stronger toward the edges
    "k3": (-1.0, 1.0),       # radial, only the far corners
    "p1": (-0.1, 0.1),       # tangential: lens tilted up/down
    "p2": (-0.1, 0.1),       # tangential: lens tilted left/right
}
STEPS = 200
WINDOW = "lens distortion"


def to_slider(name, value):
    lo, hi = RANGES[name]
    return int(round((value - lo) / (hi - lo) * STEPS))


def from_slider(name, pos):
    lo, hi = RANGES[name]
    return lo + (hi - lo) * pos / STEPS


def distort(x, y, c):
    """OpenCV's model: ideal normalized point -> where the lens actually puts it."""
    r2 = x * x + y * y
    radial = 1 + c["k1"] * r2 + c["k2"] * r2**2 + c["k3"] * r2**3
    xd = x * radial + 2 * c["p1"] * x * y + c["p2"] * (r2 + 2 * x * x)
    yd = y * radial + c["p1"] * (r2 + 2 * y * y) + 2 * c["p2"] * x * y
    return xd, yd


def undistort(xd, yd, c, iterations=20):
    """Inverse of distort(): for each camera pixel, which ideal point landed there."""
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1 + c["k1"] * r2 + c["k2"] * r2**2 + c["k3"] * r2**3
        dx = 2 * c["p1"] * x * y + c["p2"] * (r2 + 2 * x * x)
        dy = c["p1"] * (r2 + 2 * y * y) + 2 * c["p2"] * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


# Normalized coordinates of a coarse grid of camera pixels (every 4th pixel).
# The mapping is smooth, so it is solved on this grid and then upscaled: ~16x faster.
STEP = 4
u, v = np.meshgrid(np.arange(0, W + STEP, STEP), np.arange(0, H + STEP, STEP))
XD = (u - CX) / FX
YD = (v - CY) / FY


def render(c):
    # For each camera pixel, find the ideal point, then colour it by chessboard square
    x, y = undistort(XD, YD, c)
    xd, yd = distort(x, y, c)
    bad = ~(np.abs(xd - XD) + np.abs(yd - YD) < 1e-3)   # inverse did not converge
    px = np.where(bad, np.nan, x * FX + CX)              # back to ideal pixel positions
    py = np.where(bad, np.nan, y * FY + CY)
    size = (W + STEP, H + STEP)
    px = cv2.resize(px.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)[:H, :W]
    py = cv2.resize(py.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)[:H, :W]

    squares = (np.floor(px / SQUARE) + np.floor(py / SQUARE)) % 2
    img = np.where(squares == 0, 235, 30).astype(np.uint8)
    # Grey = no real point lands on this pixel (strong barrel folds back on itself)
    img[~np.isfinite(px)] = 128
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Perfect-lens grid lines for comparison
    for gx in range(0, W + 1, SQUARE):
        cv2.line(img, (gx, 0), (gx, H), (0, 0, 220), 1)
    for gy in range(0, H + 1, SQUARE):
        cv2.line(img, (0, gy), (W, gy), (0, 0, 220), 1)

    text = "  ".join(f"{k}={c[k]:+.3f}" for k in ["k1", "k2", "p1", "p2", "k3"])
    cv2.rectangle(img, (0, H - 26), (W, H), (0, 0, 0), -1)
    cv2.putText(img, text, (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def set_values(c):
    for name, value in c.items():
        cv2.setTrackbarPos(name, WINDOW, to_slider(name, value))


def main():
    cv2.namedWindow(WINDOW)
    for name in ["k1", "k2", "k3", "p1", "p2"]:
        cv2.createTrackbar(name, WINDOW, to_slider(name, 0.0), STEPS, lambda _: None)
    set_values(MINE)

    last = None
    while True:
        c = {name: from_slider(name, cv2.getTrackbarPos(name, WINDOW)) for name in RANGES}
        if c != last:                            # only re-render when a slider moved
            with np.errstate(all="ignore"):
                cv2.imshow(WINDOW, render(c))
            last = c

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            set_values(MINE)
        if key == ord("z"):
            set_values({name: 0.0 for name in RANGES})

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
