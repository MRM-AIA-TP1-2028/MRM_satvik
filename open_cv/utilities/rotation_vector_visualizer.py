"""See what a rotation vector (rvec) means.

A rotation vector packs a whole 3D rotation into 3 numbers. Its direction is
the axis to turn around and its length is how far to turn, in radians. This is
the form cv2.solvePnP() returns the marker's rotation in.

The view shows the rotation the sliders describe:
    amber arrow          the rotation axis (rvec's direction)
    grey outline/axes    the neutral pose (no rotation)
    shaded box/axes      the pose after turning by |rvec| around the axis
    coloured arcs        the path each axis tip sweeps while turning
The panel shows the numbers, the matrix cv2.Rodrigues() makes from rvec, and
the roll/pitch/yaw read from that matrix.

Run:   python rotation_vector_visualizer.py
Mouse: drag in the view to orbit, scroll to zoom, drag the sliders,
       click an example to play it, click a value to type it
Typing a value: Enter = apply, Tab = apply and go to the next, Esc = cancel.
       The deg / rad switch sets the unit, so an rvec printed by solvePnP
       (radians) can be typed in directly.
Keys:  space = replay the turn,  1-4 = examples,  0 = reset,  q / Esc = quit

The window can be resized. Text uses Pillow if it is installed (nicer fonts),
otherwise OpenCV's built-in font.
"""

import functools
import time

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

WINDOW = "Rotation vector"
START_W, START_H = 1240, 760
MIN_W, MIN_H = 900, 700      # smaller windows are scaled down instead
PANEL_W = 380
ANIM_SECONDS = 1.2
AXIS_LEN = 1.8
SLIDER_DEG = 180             # sliders run from -180 to +180 degrees

# Palette (BGR)
VIEW_TOP = (40, 33, 28)
VIEW_BOTTOM = (20, 16, 14)
GRID = (62, 54, 48)
PANEL_BG = (30, 26, 23)
CARD_BG = (40, 35, 31)
CARD_EDGE = (58, 52, 47)
TEXT = (238, 234, 230)
DIM = (150, 141, 133)
FAINT = (98, 90, 84)
ACCENT = (40, 190, 255)      # amber: the rotation axis
ACCENT_DARK = (25, 100, 140)
WARN = (115, 115, 235)
AXIS_COLOURS = [(85, 85, 240), (110, 205, 95), (245, 165, 75)]   # x red, y green, z blue

EXAMPLES = [                 # (label, rvec in degrees per component)
    ("90° z", (0, 0, 90)),
    ("90° x", (90, 0, 0)),
    ("diagonal", (29, 29, 0)),      # ~41 deg about the x-y diagonal: rvec is not 3 angles
    ("120° xyz", (69, 69, 69)),     # 120 deg about (1, 1, 1): x -> y -> z -> x
]

# A flat box, like a marker plate, so its orientation is easy to read.
# Corner index = 4*bx + 2*by + bz, where b is 0 for the -1 side and 1 for +1.
HALF = np.array([1.0, 0.7, 0.2])
BOX = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * HALF
BOX_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(BOX[i] != BOX[j]) == 1]
BOX_FACES = [([0, 1, 3, 2], 0, -1), ([4, 5, 7, 6], 0, 1),     # (corners, normal axis, sign)
             ([0, 1, 5, 4], 1, -1), ([2, 3, 7, 6], 1, 1),
             ([0, 2, 6, 4], 2, -1), ([1, 3, 7, 5], 2, 1)]

FONT_FILES = {
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "/System/Library/Fonts/Helvetica.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "mono": ["C:/Windows/Fonts/consola.ttf", "/System/Library/Fonts/Menlo.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}


# --------------------------------------------------------------------------
# Text: collected while drawing, rendered in one pass at the end
# --------------------------------------------------------------------------
def ascii_only(s):
    return s.replace("°", " deg").replace("·", "-").replace("—", "-").replace("…", "...")


class TextLayer:
    def __init__(self):
        self.items = []
        self.fonts = {}

    def font(self, size, style):
        if Image is None:
            return None
        key = (size, style)
        if key not in self.fonts:
            self.fonts[key] = None
            for path in FONT_FILES[style]:
                try:
                    self.fonts[key] = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
            if self.fonts[key] is None:
                try:
                    self.fonts[key] = ImageFont.load_default(size=size)
                except TypeError:            # Pillow < 10.1
                    self.fonts[key] = ImageFont.load_default()
        return self.fonts[key]

    def width(self, s, size, style="regular"):
        f = self.font(size, style)
        if f is None:
            return cv2.getTextSize(ascii_only(s), cv2.FONT_HERSHEY_SIMPLEX, size / 30, 1)[0][0]
        return f.getlength(s)

    def add(self, s, x, y, size=14, colour=TEXT, style="regular", align="l"):
        """Queue text with its vertical centre at y; align is l, m or r."""
        self.items.append((s, float(x), float(y), size, colour, style, align))

    def render(self, img):
        if Image is None:
            for s, x, y, size, colour, style, align in self.items:
                s = ascii_only(s)
                scale = size / 30
                (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                x -= {"l": 0, "m": w / 2, "r": w}[align]
                cv2.putText(img, s, (int(x), int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                            colour, 2 if style == "bold" else 1, cv2.LINE_AA)
        else:
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            for s, x, y, size, colour, style, align in self.items:
                draw.text((x, y), s, font=self.font(size, style), fill=colour, anchor=align + "m")
            img[:] = np.asarray(pil)
        self.items.clear()


# --------------------------------------------------------------------------
# Small drawing helpers
# --------------------------------------------------------------------------
def pt(p):
    return int(round(p[0])), int(round(p[1]))


def line(img, p, q, colour, thickness=1):
    cv2.line(img, pt(p), pt(q), colour, thickness, cv2.LINE_AA)


def shade(colour, k):
    return tuple(int(c * k) for c in colour)


def rounded_rect(img, x0, y0, x1, y1, r, fill, edge=None):
    x0, y0, x1, y1, r = (int(round(v)) for v in (x0, y0, x1, y1, r))
    if edge is not None:
        rounded_rect(img, x0, y0, x1, y1, r, edge)
        x0, y0, x1, y1, r = x0 + 1, y0 + 1, x1 - 1, y1 - 1, max(r - 1, 0)
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), fill, -1)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), fill, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (cx, cy), r, fill, -1, cv2.LINE_AA)


def arrowhead(img, p_from, p_tip, colour, size=10):
    d = np.asarray(p_tip, float) - p_from
    n = np.linalg.norm(d)
    if n < 1e-3:
        return
    d /= n
    side = np.array([-d[1], d[0]]) * size * 0.5
    tri = np.array([p_tip + d * size * 0.3, p_tip - d * size + side, p_tip - d * size - side])
    cv2.fillConvexPoly(img, np.round(tri).astype(np.int32), colour, cv2.LINE_AA)


@functools.lru_cache(maxsize=4)
def gradient(h, w):
    t = np.linspace(0, 1, h)[:, None, None]
    col = (1 - t) * np.array(VIEW_TOP) + t * np.array(VIEW_BOTTOM)
    return np.repeat(col, w, axis=1).astype(np.uint8)


# --------------------------------------------------------------------------
# Rotation maths
# --------------------------------------------------------------------------
def rotation_to_euler(R):
    """Rotation matrix -> (roll, pitch, yaw) in degrees, as in aruco_medium.py."""
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                                    # gimbal lock: pitch = +-90 deg
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


class Camera:
    """Virtual pinhole camera orbiting the origin, world z up."""

    def __init__(self, azimuth, elevation, distance, width, height):
        eye = distance * np.array([np.cos(elevation) * np.cos(azimuth),
                                   np.cos(elevation) * np.sin(azimuth),
                                   np.sin(elevation)])
        forward = -eye / distance
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.R = np.array([right, down, forward])
        self.t = -self.R @ eye
        self.f = 1.1 * min(width, height)
        self.c = np.array([width / 2, height / 2])

    def to_cam(self, points):
        return np.asarray(points, float).reshape(-1, 3) @ self.R.T + self.t

    def project(self, points):
        p = self.to_cam(points)
        return self.f * p[:, :2] / p[:, 2:3] + self.c


# --------------------------------------------------------------------------
# 3D view
# --------------------------------------------------------------------------
def draw_grid(img, cam, z=-1.5, extent=3.0, step=0.5):
    for v in np.arange(-extent, extent + 1e-9, step):
        colour = shade(GRID, 1 - 0.7 * abs(v) / extent)
        for a, b in (((v, -extent, z), (v, extent, z)), ((-extent, v, z), (extent, v, z))):
            p, q = cam.project([a, b])
            line(img, p, q, colour)


def draw_axes(img, cam, R, colours, thickness, text=None):
    origin, *tips = cam.project(np.vstack([np.zeros(3), AXIS_LEN * R.T]))
    for tip, colour, name in zip(tips, colours, "xyz"):
        line(img, origin, tip, colour, thickness)
        if text is not None:
            d = tip - origin
            d = d / max(np.linalg.norm(d), 1e-6)
            text.add(name, *(tip + 16 * d), 17, colour, "bold", "m")


def draw_box(img, cam, R):
    """Shaded, semi-transparent box: back edges, then front faces, then front edges."""
    world = BOX @ R.T
    cam_pts = cam.to_cam(world)
    px = cam.project(world)

    front = []
    for corners, ax, sign in BOX_FACES:
        normal = cam.R @ (sign * R[:, ax])
        centre = cam_pts[corners].mean(axis=0)
        facing = -centre @ normal / np.linalg.norm(centre)
        if facing > 0:
            front.append((corners, ax, facing))

    front_corners = [set(c) for c, _, _ in front]
    is_front = [any(i in s and j in s for s in front_corners) for i, j in BOX_EDGES]
    for (i, j), f in zip(BOX_EDGES, is_front):
        if not f:
            line(img, px[i], px[j], shade(TEXT, 0.35))

    overlay = img.copy()
    for corners, ax, facing in front:
        poly = np.round(px[corners]).astype(np.int32)
        cv2.fillPoly(overlay, [poly], shade(AXIS_COLOURS[ax], 0.35 + 0.65 * facing), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)

    for (i, j), f in zip(BOX_EDGES, is_front):
        if f:
            line(img, px[i], px[j], shade(TEXT, 0.9), 2)


def draw_view(rvec, frac, cam, w, h, text):
    img = gradient(h, w).copy()
    draw_grid(img, cam)

    neutral = np.eye(3)
    for i, j in BOX_EDGES:
        p = cam.project(BOX[[i, j]])
        line(img, p[0], p[1], FAINT)
    draw_axes(img, cam, neutral, [FAINT] * 3, 1)

    angle = np.linalg.norm(rvec)
    if angle > 1e-6:
        axis = rvec / angle
        a, b = cam.project([-2.6 * axis, 2.6 * axis])
        line(img, a, b, ACCENT_DARK)
        origin, tip = cam.project([np.zeros(3), 2.6 * axis])
        line(img, origin, tip, ACCENT, 2)
        arrowhead(img, origin, tip, ACCENT, 14)
        d = (tip - origin) / max(np.linalg.norm(tip - origin), 1e-6)
        text.add("rotation axis", *(tip + 22 * d), 14, ACCENT, "regular", "m")

        if frac > 0.01:
            for e, colour in zip(np.eye(3) * AXIS_LEN, AXIS_COLOURS):
                arc = [cv2.Rodrigues(axis * t)[0] @ e for t in np.linspace(0, angle * frac, 64)]
                arc_px = cam.project(arc)
                cv2.polylines(img, [np.round(arc_px).astype(np.int32)], False, colour, 1, cv2.LINE_AA)
                arrowhead(img, arc_px[-6], arc_px[-1], colour, 10)

    R, _ = cv2.Rodrigues(rvec * frac)
    draw_box(img, cam, R)
    draw_axes(img, cam, R, AXIS_COLOURS, 3, text)

    text.add("grey = neutral pose    colour = after the turn", 20, 24, 13, DIM)
    if frac < 1:
        text.add(f"turning…  {np.degrees(angle * frac):5.1f}° of {np.degrees(angle):.1f}°",
                 20, h - 26, 15, ACCENT)
    else:
        text.add("drag to orbit  ·  scroll to zoom  ·  q to quit", 20, h - 26, 13, FAINT)
    return img


# --------------------------------------------------------------------------
# Side panel
# --------------------------------------------------------------------------
def format_value(deg, unit):
    if unit == "rad":
        return f"{np.radians(deg):+.3f}"
    return f"{deg:+.0f}°" if float(deg).is_integer() else f"{deg:+.1f}°"


def plain_value(deg, unit):
    """The value as it appears in the box when editing starts."""
    return f"{np.radians(deg):.3f}" if unit == "rad" else f"{deg:g}"


def draw_panel(state, rvec, h, x_off, text):
    """Draw the panel and return it with its clickable regions (in window coords)."""
    img = np.full((h, PANEL_W, 3), PANEL_BG, np.uint8)
    cv2.line(img, (0, 0), (0, h), CARD_EDGE, 1)
    hits = []
    mx, my = state["mouse"]
    pad = 22
    x0, x1 = pad, PANEL_W - pad

    def hovered(r):
        return r[0] <= mx - x_off <= r[2] and r[1] <= my <= r[3]

    def label(s, x, y, size=14, colour=TEXT, style="regular", align="l"):
        text.add(s, x_off + x, y, size, colour, style, align)

    y = 36
    label("Rotation vector", x0, y, 24, TEXT, "bold")
    y += 26
    label("one turn, around one axis", x0, y, 14, DIM)

    def chip_width(s, size=13, height=28):
        return text.width(s, size) + 0.85 * height

    def chip(x, s, action, selected=False, size=13, height=28):
        w = chip_width(s, size, height)
        r = (x, y - height / 2, x + w, y + height / 2)
        edge = ACCENT if selected else (shade(TEXT, 0.5) if hovered(r) else CARD_EDGE)
        rounded_rect(img, *r, height / 2, CARD_BG, edge)
        label(s, x + w / 2, y, size, ACCENT if selected else TEXT, "regular", "m")
        hits.append(((x_off + r[0], r[1], x_off + r[2], r[3]), action))
        return x + w + 8

    # Sliders with a value box each, and the unit switch
    unit = state["unit"]
    y += 34
    label("RVEC  ·  click a value to type it", x0, y, 11, FAINT, "bold")
    w_rad = chip_width("rad", 12, 22)
    chip(x1 - w_rad, "rad", ("unit", "rad"), unit == "rad", 12, 22)
    chip(x1 - w_rad - 6 - chip_width("deg", 12, 22), "deg", ("unit", "deg"), unit == "deg", 12, 22)
    y += 32
    box_w = 72
    tx0, tx1 = x0 + 26, x1 - box_w - 18
    mid = (tx0 + tx1) / 2
    edit = state["edit"]
    for i, name in enumerate("xyz"):
        v = state["deg"][i]
        colour = AXIS_COLOURS[i]
        kx = mid + (tx1 - tx0) / 2 * v / SLIDER_DEG
        label(name, x0, y, 17, colour, "bold")
        cv2.line(img, (tx0, y), (tx1, y), CARD_EDGE, 4, cv2.LINE_AA)
        cv2.line(img, (int(mid), y - 6), (int(mid), y + 6), FAINT, 1, cv2.LINE_AA)
        cv2.line(img, (int(mid), y), (int(kx), y), colour, 4, cv2.LINE_AA)
        active = state["drag"] and state["drag"][0] == "slider" and state["drag"][1] == i
        r = (tx0 - 10, y - 14, tx1 + 10, y + 14)
        cv2.circle(img, (int(kx), y), 9 if active or hovered(r) else 8, TEXT, -1, cv2.LINE_AA)
        cv2.circle(img, (int(kx), y), 9 if active or hovered(r) else 8, colour, 2, cv2.LINE_AA)
        hits.append(((x_off + r[0], r[1], x_off + r[2], r[3]), ("slider", i, x_off + tx0, x_off + tx1)))

        # Value box: click to type
        box = (x1 - box_w, y - 13, x1, y + 13)
        editing = edit is not None and edit["i"] == i
        edge = ACCENT if editing else (shade(TEXT, 0.5) if hovered(box) else CARD_EDGE)
        rounded_rect(img, *box, 6, CARD_BG, edge)
        tx = x1 - 9
        if editing:
            s = edit["buf"]
            sw = text.width(s, 14, "mono")
            if edit["fresh"] and s:          # whole value selected: typing replaces it
                cv2.rectangle(img, (int(tx - sw - 2), y - 9), (int(tx + 1), y + 9), ACCENT_DARK, -1)
            elif int(time.perf_counter() * 2) % 2 == 0:
                cv2.line(img, (tx + 2, y - 8), (tx + 2, y + 8), TEXT, 1)
            label(s, tx, y, 14, TEXT, "mono", "r")
        else:
            label(format_value(v, unit), tx, y, 14, TEXT, "mono", "r")
        hits.append(((x_off + box[0], box[1], x_off + box[2], box[3]), ("field", i)))
        y += 36

    # Example chips and buttons
    y += 4
    label("EXAMPLES", x0, y, 11, FAINT, "bold")
    y += 26

    x = x0
    for s, values in EXAMPLES:
        x = chip(x, s, ("example", values), tuple(state["deg"]) == values)
    y += 38
    x = chip(x0, "Replay  (space)", ("replay",))
    chip(x, "Reset  (0)", ("reset",))

    # Card: angle and axis
    y += 30
    angle = np.linalg.norm(rvec)
    rounded_rect(img, x0, y, x1, y + 78, 10, CARD_BG, CARD_EDGE)
    label("angle = |rvec|", x0 + 16, y + 20, 12, DIM)
    label(f"{np.degrees(angle):.1f}°", x0 + 16, y + 50, 26, TEXT, "bold")
    label(f"{angle:.3f} rad", x0 + 16 + text.width(f"{np.degrees(angle):.1f}°", 26, "bold") + 10,
          y + 54, 13, DIM)
    label("axis = rvec / |rvec|", x1 - 16, y + 20, 12, DIM, "regular", "r")
    axis_s = ("(" + ", ".join(f"{v:.2f}" for v in rvec / angle) + ")") if angle > 1e-6 else "none"
    label(axis_s, x1 - 16, y + 50, 13, ACCENT, "mono", "r")

    # Card: rotation matrix
    y += 88
    R, _ = cv2.Rodrigues(rvec)
    R[np.abs(R) < 0.005] = 0.0               # avoid printing "-0.00"
    rounded_rect(img, x0, y, x1, y + 136, 10, CARD_BG, CARD_EDGE)
    label("R = cv2.Rodrigues(rvec)", x0 + 16, y + 20, 12, DIM)
    label("column = where that axis ends up", x1 - 16, y + 20, 11, FAINT, "regular", "r")
    col_x = [x0 + 100, x0 + 190, x0 + 280]
    for r_i in range(3):
        for c in range(3):
            label(f"{R[r_i, c]:.2f}", col_x[c], y + 48 + 24 * r_i, 16, AXIS_COLOURS[c], "mono", "r")
    for c, name in enumerate("xyz"):
        label(name, col_x[c] - 18, y + 120, 13, AXIS_COLOURS[c], "bold", "m")

    # Card: angles
    y += 146
    rounded_rect(img, x0, y, x1, y + 100, 10, CARD_BG, CARD_EDGE)
    label("roll · pitch · yaw  from R", x0 + 16, y + 20, 12, DIM)
    for c, v in enumerate(rotation_to_euler(R)):
        label(f"{round(v, 1) + 0.0:.1f}°", col_x[c], y + 44, 16, TEXT, "mono", "r")   # + 0.0: no "-0.0"
    label("np.degrees(rvec) — not angles!", x0 + 16, y + 68, 12, WARN)
    for c, v in enumerate(np.degrees(rvec)):
        label(f"{round(v, 1) + 0.0:.1f}°", col_x[c], y + 88, 16, WARN, "mono", "r")
    return img, hits


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def anim_fraction(state):
    t = min(max((time.perf_counter() - state["anim_start"]) / ANIM_SECONDS, 0.0), 1.0)
    return t * t * (3 - 2 * t)               # ease in and out


def start_edit(state, i):
    state["edit"] = {"i": i, "buf": plain_value(state["deg"][i], state["unit"]), "fresh": True}


def commit_edit(state):
    """Apply the typed value (ignored if it is not a number) and play the turn."""
    edit, state["edit"] = state["edit"], None
    try:
        v = float(edit["buf"])
    except ValueError:
        return
    deg = np.degrees(v) if state["unit"] == "rad" else v
    state["deg"][edit["i"]] = float(np.clip(deg, -SLIDER_DEG, SLIDER_DEG))
    state["anim_start"] = time.perf_counter()


def edit_key(state, key):
    edit = state["edit"]
    if key in (10, 13):                      # Enter
        commit_edit(state)
    elif key == 27:                          # Esc
        state["edit"] = None
    elif key == 9:                           # Tab
        commit_edit(state)
        start_edit(state, (edit["i"] + 1) % 3)
    elif key == 8:                           # Backspace
        edit["buf"] = "" if edit["fresh"] else edit["buf"][:-1]
        edit["fresh"] = False
    elif chr(key) in "0123456789.-+":
        edit["buf"] = (chr(key) if edit["fresh"] else edit["buf"] + chr(key))[:10]
        edit["fresh"] = False


def render(state, w, h, text):
    rvec = np.radians(np.array(state["deg"], float))
    view_w = w - PANEL_W
    cam = Camera(state["az"], state["el"], state["dist"], view_w, h)
    view = draw_view(rvec, anim_fraction(state), cam, view_w, h, text)
    panel, hits = draw_panel(state, rvec, h, view_w, text)
    frame = np.hstack([view, panel])
    text.render(frame)
    return frame, hits, view_w


def main():
    state = {
        "deg": list(EXAMPLES[0][1]),
        "az": np.radians(35), "el": np.radians(25), "dist": 7.0,
        "anim_start": time.perf_counter(),
        "drag": None, "mouse": (-1, -1), "hits": [], "view_w": START_W - PANEL_W,
        "unit": "deg", "edit": None,
    }

    def play(values=None):
        if values is not None:
            state["deg"] = list(values)
        state["anim_start"] = time.perf_counter()

    def set_slider(i, x, tx0, tx1):
        v = round(((x - tx0) / (tx1 - tx0) * 2 - 1) * SLIDER_DEG)
        state["deg"][i] = int(np.clip(v, -SLIDER_DEG, SLIDER_DEG))
        state["anim_start"] = -np.inf        # show the result straight away

    def on_mouse(event, x, y, flags, _):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            target = next((a for (rx0, ry0, rx1, ry1), a in state["hits"]
                           if rx0 <= x <= rx1 and ry0 <= y <= ry1), None)
            if state["edit"] and target != ("field", state["edit"]["i"]):
                commit_edit(state)           # clicking elsewhere applies the typed value
            for (rx0, ry0, rx1, ry1), action in state["hits"]:
                if rx0 <= x <= rx1 and ry0 <= y <= ry1:
                    if action[0] == "field":
                        if not state["edit"]:
                            start_edit(state, action[1])
                    elif action[0] == "unit":
                        state["unit"] = action[1]
                    elif action[0] == "slider":
                        state["drag"] = action
                        set_slider(action[1], x, *action[2:])
                    elif action[0] == "example":
                        play(action[1])
                    elif action[0] == "replay":
                        play()
                    elif action[0] == "reset":
                        state["deg"] = [0, 0, 0]
                    return
            if x < state["view_w"]:
                state["drag"] = ("orbit", x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            if state["drag"][0] == "slider":
                set_slider(state["drag"][1], x, *state["drag"][2:])
            else:
                _, px, py = state["drag"]
                state["az"] -= (x - px) * 0.01
                state["el"] = float(np.clip(state["el"] + (y - py) * 0.01, -1.45, 1.45))
                state["drag"] = ("orbit", x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom = 0.9 if cv2.getMouseWheelDelta(flags) > 0 else 1.1
            state["dist"] = float(np.clip(state["dist"] * zoom, 5.0, 14.0))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, START_W, START_H)
    cv2.setMouseCallback(WINDOW, on_mouse)
    text = TextLayer()
    keys = {ord(str(n + 1)): values for n, (_, values) in enumerate(EXAMPLES)}

    while True:
        _, _, w, h = cv2.getWindowImageRect(WINDOW)
        frame, state["hits"], state["view_w"] = render(state, max(w, MIN_W), max(h, MIN_H), text)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(15) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if state["edit"]:
            if key != 255:
                edit_key(state, key)
            continue
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            play()
        elif key == ord("0"):
            state["deg"] = [0, 0, 0]
        elif key in keys:
            play(keys[key])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
