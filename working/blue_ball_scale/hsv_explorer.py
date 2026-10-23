"""HSV explorer: see what H, S and V mean, and find colour ranges for tracker.py.

    python hsv_explorer.py

Window layout
    top row     camera | mask (white = inside the range) | only the matching pixels
    bottom      hue x saturation chart at the current V max; the white box is your range

Sliders     H min/max (0-179), S min/max (0-255), V min/max (0-255)
            If H min > H max the range wraps around 179 -> 0 (needed for red).
Mouse       click the camera panel to read that pixel's H, S, V
Keys        p  print the current range as code to paste into tracker.py
            q  quit

If no camera opens, a test picture is used instead: two colour wheels, one
bright and one "in shadow", so you can see that shadow changes V, not H.
"""

import cv2
import numpy as np

SOURCE = 0                 # 0 = MacBook camera
MIRROR = True              # flip the camera left-right like a mirror
W, H = 480, 270            # size of each top panel
CHART_H = 150              # height of the hue x saturation chart
WIN = "HSV explorer"
START = {"H min": 22, "H max": 43, "S min": 64, "S max": 255, "V min": 127, "V max": 255}  # yellow-green (tennis ball)


def test_picture():
    """Two colour wheels: hue = angle, saturation = distance from centre.
    Left wheel bright (V = 255), right wheel in 'shadow' (V = 110)."""
    yy, xx = np.mgrid[0:H, 0:W // 2]
    cx, cy, r = W // 4, H // 2, min(W // 4, H // 2) - 10
    dx, dy = xx - cx, yy - cy
    dist = np.hypot(dx, dy)
    hue = ((np.degrees(np.arctan2(-dy, dx)) % 360) / 2).astype(np.uint8)
    sat = np.clip(dist / r * 255, 0, 255).astype(np.uint8)
    halves = []
    for v in (255, 110):
        hsv = np.dstack([hue, sat, np.full_like(hue, v)])
        hsv[dist > r] = (0, 0, 40)              # dark grey outside the wheel
        halves.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    return np.hstack(halves)


def make_mask(hsv, lo, hi):
    """inRange, but a hue range with H min > H max wraps around (red)."""
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    upper = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]], np.uint8), np.array([179, hi[1], hi[2]], np.uint8))
    lower = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), np.array([hi[0], hi[1], hi[2]], np.uint8))
    return cv2.bitwise_or(upper, lower)


def hs_chart(lo, hi, width):
    """Every hue (left to right) x every saturation (top 255 -> bottom 0),
    all drawn at brightness V max. The white box is the selected H and S range."""
    lo, hi = [int(v) for v in lo], [int(v) for v in hi]   # plain ints: uint8 maths would overflow
    hsv = np.zeros((CHART_H, width, 3), np.uint8)
    hsv[..., 0] = np.linspace(0, 179, width).astype(np.uint8)[None, :]
    hsv[..., 1] = np.linspace(255, 0, CHART_H).astype(np.uint8)[:, None]
    hsv[..., 2] = hi[2]
    chart = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def x(h):
        return int(h * (width - 1) / 179)

    y0, y1 = int((255 - hi[1]) * (CHART_H - 1) / 255), int((255 - lo[1]) * (CHART_H - 1) / 255)
    spans = [(lo[0], hi[0])] if lo[0] <= hi[0] else [(lo[0], 179), (0, hi[0])]
    for h0, h1 in spans:
        cv2.rectangle(chart, (x(h0), y0), (x(h1), y1), (255, 255, 255), 2)
    for h, name in [(0, "red"), (30, "yellow"), (60, "green"), (90, "cyan"), (120, "blue"), (150, "magenta")]:
        label(chart, f"{name} {h}", (x(h) + 4, CHART_H - 8), 0.45)
    label(chart, "S 255", (4, 16), 0.45)
    label(chart, f"hue -> (chart drawn at V = {hi[2]})", (width - 300, 16), 0.45)
    return chart


def label(img, text, org, size=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(SOURCE)
    use_camera = cap.isOpened() and cap.read()[0]
    still = None if use_camera else test_picture()
    if not use_camera:
        print("No camera - showing the test picture instead.")

    cv2.namedWindow(WIN)
    for name, value in START.items():
        cv2.createTrackbar(name, WIN, value, 179 if name.startswith("H") else 255, lambda _: None)

    clicked = {"text": "click the camera panel to read a pixel"}
    hsv = None

    def on_mouse(event, mx, my, *_):
        if event == cv2.EVENT_LBUTTONDOWN and mx < W and my < H and hsv is not None:
            h, s, v = hsv[my, mx]
            clicked["text"] = f"pixel ({mx},{my}):  H={h}  S={s}  V={v}"
            print(clicked["text"])

    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        if use_camera:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H))
            if MIRROR:
                frame = cv2.flip(frame, 1)
        else:
            frame = still.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        t = {name: cv2.getTrackbarPos(name, WIN) for name in START}
        lo = np.array([t["H min"], t["S min"], t["V min"]], np.uint8)
        hi = np.array([t["H max"], t["S max"], t["V max"]], np.uint8)
        mask = make_mask(hsv, lo, hi)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        panels = [frame.copy(), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result]
        for img, title in zip(panels, ["camera", "mask", "matching pixels"]):
            label(img, title, (8, 22))
        label(panels[0], clicked["text"], (8, H - 10), 0.45)
        top = np.hstack(panels)
        cv2.imshow(WIN, np.vstack([top, hs_chart(lo, hi, top.shape[1])]))

        key = cv2.waitKey(30) & 0xFF
        if key == ord("p"):
            print(f"(np.array([{lo[0]}, {lo[1]}, {lo[2]}]), np.array([{hi[0]}, {hi[1]}, {hi[2]}]))")
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
