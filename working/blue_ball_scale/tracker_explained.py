"""Blue object + tennis ball detection and tracking with ArUco scale estimation.

Live camera stream -> detects every blue object (any shape) and a tennis ball,
tracks each blue object over time, and reports sizes in cm using an ArUco
marker of known side length as the scale reference. With no marker in view,
the tennis ball's known diameter (BALL_CM) is used instead.

Windows
    Tracker - left: camera with contours, bounding boxes, center trails, ball
              circle, labels and distances between blue objects
              right: black panel with (x, y) of each blue object, its position
              relative to the ball and distance traversed
    Masks   - (toggle with 'm') the blue and ball HSV masks

Keys
    q / Esc  quit
    c        clear trails and traversed distances
    r        reset the blue HSV range to the default
    m        toggle mask windows
    click    on a blue object in the camera view to select its colour
"""

# =============================================================================
# tracker_explained.py
# -----------------------------------------------------------------------------
# This is EXACTLY the same program as tracker.py, line for line. The only
# difference is the comments. Run either one; they behave identically.
#
# BIG PICTURE (read this first)
#   1. Open the webcam.
#   2. For every frame:
#        a. Blur the image a little and convert it to HSV colour space.
#        b. Look for an ArUco marker (4x4, IDs 0-49). Its known real size tells us how many
#           pixels make one centimetre (the "scale").
#        c. Find blue blobs (any shape) using a colour range on hue.
#        d. Find the tennis ball: a yellow-green blob that is also round.
#        e. Match this frame's blue blobs to last frame's so each keeps an ID,
#           a trail, and a running total of distance travelled.
#        f. Draw everything in one "Tracker" window: camera view + black text panel.
#   3. Read the keyboard; repeat until the user quits.
#
# The string at the very top (in triple quotes) is the module "docstring". It
# is not code that runs; it is documentation for whoever reads the file.
# =============================================================================


# --- Imports -----------------------------------------------------------------
# math: square roots, pi, distances (math.dist, math.hypot)
import itertools
import math
# deque: a list with a maximum length. When full, adding a new item pushes the
# oldest one out. Perfect for a trail of "the last 64 positions".
from collections import deque

# cv2 is OpenCV: camera access, colour conversion, contours, drawing, ArUco.
import cv2
# numpy: fast arrays. OpenCV images ARE numpy arrays (height x width x 3).
import numpy as np


# --- Constants (settings that never change while the program runs) ----------

# Settings: edit these instead of passing command-line options
# Which camera to open. 0 = the MacBook Air's built-in camera. If your iPhone
# is nearby, macOS Continuity Camera may add it as another camera (usually 1).
# You can also put a video file path here, e.g. "clip.mp4".
SOURCE = 0          # 0 = MacBook Air camera (an iPhone via Continuity Camera may be 1)
# Measure your printed marker's black square with a ruler and put it here.
# Every cm value in the program depends on this number being right.
MARKER_CM = 5.0     # side length of the printed ArUco marker's black square in cm
# The tennis ball is 6.6 cm (66 mm) across. When no marker is visible, the ball's
# diameter in pixels / 6.6 gives the scale instead (less accurate: the radius
# jitters, and it only holds for things as far from the camera as the ball).
BALL_CM = 6.6       # tennis ball diameter in cm
# Blobs smaller than these (in square pixels) are ignored as noise.
MIN_BLUE_AREA = 600  # min blue blob area in px^2
MIN_BALL_AREA = 300  # min ball blob area in px^2
# The camera resolution we ask for. The M4 MacBook Air's camera can record
# 1920x1080, but at that size the Tracker window (1920 + 380 wide) doesn't fit
# on the 15-inch screen (1710 points wide), and every frame takes longer to
# process. 1280x720 (1280 + 380 = 1660 wide) fits and is fast. If you change it, scale
# the minimum areas too: they are in px^2, so 1080p needs (1920/1280)^2 = 2.25x.
FRAME_W, FRAME_H = 1280, 720  # camera can do 1920x1080; 720p fits the screen and runs faster
MIRROR = True      # flip the picture left-right so it moves like a mirror

# The ArUco marker family we search for. "4X4" means a 4x4 grid of black and
# white bits inside the marker's black border. "_50" means the dictionary has
# 50 marker IDs (0-49). Only one dictionary, and a small one, because:
#   - we only need ONE marker of known size; its ID doesn't matter
#   - fewer valid patterns = fewer random textures mistaken for a marker,
#     and more misread bits can be corrected
#   - one search per frame instead of three is faster
# make_marker.py prints marker ID 0 from this same dictionary.
ARUCO_DICT = cv2.aruco.DICT_4X4_50  # make_marker.py prints from this dictionary

# Colour ranges in HSV.
#   H (hue) = the colour itself. OpenCV stores it as 0-179 (half of 0-360 deg).
#   S (saturation) = how strong the colour is (0 = grey, 255 = vivid).
#   V (value) = brightness (0 = black, 255 = bright).
# OpenCV HSV: H in [0, 179], S and V in [0, 255]
# Blue: hue 100-130. Saturation at least 120 so greyish pixels are ignored.
# Value at least 50 so very dark pixels are ignored.
DEFAULT_BLUE = (np.array([100, 120, 50]), np.array([130, 255, 255]))
# Tennis-ball yellow-green: hue 22-43.
BALL_RANGE = (np.array([22, 64, 127]), np.array([43, 255, 255]))
# A 5x5 ellipse-shaped "kernel" (a small stamp) used by morphology to clean
# masks. Ellipse instead of square so it doesn't make blobs look boxy.
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Drawing colours. NOTE: OpenCV uses BGR order (blue, green, red), not RGB.
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
BLUE_OUTLINE = (255, 140, 0)   # B=255, G=140, R=0 -> strong blue + some green = azure
BOX_COLOR = (0, 255, 255)      # G + R = yellow
BALL_COLOR = (0, 255, 0)       # pure green
LINK_COLOR = (0, 0, 255)       # pure red (line from blue object to ball)


# =============================================================================
# SCALE: turning pixels into centimetres
# =============================================================================
class ScaleEstimator:
    """Pixels-per-cm from any visible ArUco marker, else from the ball's diameter."""

    # __init__ runs once, when we write ScaleEstimator(5.0).
    # marker_cm = the real side length of the printed marker's black square.
    # alpha = how much each new measurement counts when smoothing (0.3 = 30%).
    def __init__(self, marker_cm, alpha=0.3):
        # One detector for our dictionary. DetectorParameters() = default
        # detection settings, which work fine for a printed marker.
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
        # Store the real size so update() can use it.
        self.marker_cm = marker_cm
        self.alpha = alpha  # smoothing factor for the running estimate
        # No scale yet: None means "we have never seen a marker".
        self.px_per_cm = None
        # Where this frame's scale came from (for the on-screen text):
        #   "marker" - a marker is visible
        #   "ball"   - no marker, but the ball is visible
        #   "held"   - neither visible, keeping the last value
        self.source = None

    # Called once per frame with the greyscale image (ArUco works on grey) and
    # the ball found this frame (None if there isn't one).
    def update(self, gray, ball=None):
        # detectMarkers returns:
        #   corners  - for each marker, its 4 corner points in pixels
        #   ids      - the marker numbers (or None if nothing found)
        #   _        - rejected candidates (we don't need them, hence "_")
        # A mirrored marker is not a valid code, so look for markers in the
        # un-mirrored picture and then mirror their corners back.
        if MIRROR:
            gray = cv2.flip(gray, 1)
        corners, ids, _ = self.detector.detectMarkers(gray)
        # Turn the result into a plain list of markers; empty if none found.
        corners_all = list(corners) if ids is not None else []
        if MIRROR:
            for c in corners_all:
                c[..., 0] = gray.shape[1] - 1 - c[..., 0]

        if corners_all:
            sides = []
            for c in corners_all:
                # Each marker's corners come as shape (1, 4, 2); reshape to
                # 4 rows of (x, y) so they are easy to work with.
                pts = c.reshape(4, 2)
                # np.roll shifts the rows by one: corner 0,1,2,3 -> 1,2,3,0.
                # Subtracting gives the 4 edge vectors (corner to next corner).
                # np.linalg.norm(..., axis=1) gives each edge's length in px.
                sides.extend(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1))
            # Average side length in pixels / real side in cm = pixels per cm.
            # Example: marker is 100 px wide and really 5 cm -> 20 px per cm.
            ppc = float(np.mean(sides)) / self.marker_cm
            self.source = "marker"
        elif ball:
            # No marker: the ball's diameter (2 x radius) in px / its real
            # diameter in cm. Example: ball 132 px across / 6.6 cm -> 20 px per cm.
            ppc = 2 * ball["radius"] / BALL_CM
            self.source = "ball"
        else:
            # Nothing to measure this frame.
            ppc = None
            self.source = "held" if self.px_per_cm else None

        if ppc:
            if self.px_per_cm is None:
                # First ever measurement: just take it.
                self.px_per_cm = ppc
            else:
                # Exponential moving average: 70% old value + 30% new value.
                # Stops the cm numbers on screen from flickering every frame.
                self.px_per_cm = (1 - self.alpha) * self.px_per_cm + self.alpha * ppc
        # If neither was found, px_per_cm is left unchanged, so the last
        # good scale is "held" until a marker or the ball is seen again.
        # The corners are returned so the camera view can outline the marker.
        return corners_all

    def length(self, px):
        """Format a pixel length in cm when the scale is known, else in px."""
        if self.px_per_cm:
            # Lengths divide by px_per_cm. :.1f = one decimal place.
            return f"{px / self.px_per_cm:.1f} cm"
        # No marker seen yet: show raw pixels with no decimals.
        return f"{px:.0f} px"

    def area(self, area_px):
        if self.px_per_cm:
            # Areas divide by px_per_cm SQUARED, because both width and
            # height shrink. 20 px/cm -> 400 px^2 per cm^2.
            return f"{area_px / self.px_per_cm ** 2:.1f} cm^2"
        return f"{area_px:.0f} px^2"


# =============================================================================
# DETECTION HELPERS
# =============================================================================
def clean_mask(mask):
    # A mask is a black/white image: white (255) where the colour matched.
    # OPEN = erode then dilate. Erosion eats away tiny white specks (noise);
    # dilation grows the surviving blobs back to their size.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    # CLOSE = dilate then erode. Fills small black holes inside blobs (for
    # example a reflection on the object). Done twice (iterations=2).
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)


def centroid(contour):
    # Image moments are weighted sums over the shape's pixels.
    #   m00 = area, m10 = sum of x, m01 = sum of y.
    m = cv2.moments(contour)
    if m["m00"] == 0:
        # Zero area: can't divide by zero, so report "no centre".
        return None
    # Average x = m10 / m00, average y = m01 / m00 -> the centre of the shape.
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def find_blue(hsv, lo, hi, min_area):
    """Every blue blob above min_area; no assumption about its shape."""
    # inRange: pixel becomes 255 (white) if its H, S and V are all between lo
    # and hi, else 0 (black). Then clean it up.
    mask = clean_mask(cv2.inRange(hsv, lo, hi))
    # findContours traces the outline of every white blob.
    #   RETR_EXTERNAL       - only outer outlines (ignore holes inside blobs)
    #   CHAIN_APPROX_SIMPLE - store only corner points of straight edges (saves memory)
    # The second return value is the hierarchy, which we don't need.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for c in contours:
        # Area inside the outline in px^2. Follows the real shape, so it is
        # correct for any shape (not just rectangles or circles).
        area = cv2.contourArea(c)
        if area < min_area:
            # Too small: probably noise or a far-away speck. Skip it.
            continue
        center = centroid(c)
        if center is None:
            continue
        # Store everything about this blob in a dictionary.
        objects.append({"contour": c, "area_px": area, "center": center})
    # Return the mask too, so the 'm' key can display it for debugging.
    return objects, mask


def find_ball(hsv, min_area, min_circularity=0.6, min_fill=0.5):
    """The most ball-like yellow-green blob, or None."""
    # Same idea as find_blue, but with the fixed yellow-green range.
    # The * "unpacks" BALL_RANGE (lo, hi) into two separate arguments.
    mask = clean_mask(cv2.inRange(hsv, *BALL_RANGE))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Keep track of the best candidate so far.
    best, best_score = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        # arcLength = perimeter of the outline. True = the outline is closed.
        perimeter = cv2.arcLength(c, True)
        if area < min_area or perimeter == 0:
            continue
        # Circularity = 4*pi*A / P^2. A perfect circle gives 1.0; a long thin
        # or jagged shape gives a small number. This rejects yellow objects
        # that aren't round.
        circularity = 4 * math.pi * area / perimeter ** 2
        # Smallest circle that fits around the blob: centre (x, y) and radius r.
        (x, y), r = cv2.minEnclosingCircle(c)
        # Fill = how much of that circle the blob actually covers.
        # A ball fills its circle; a crescent or an "L" shape does not.
        fill = area / (math.pi * r * r)
        if circularity < min_circularity or fill < min_fill:
            continue
        # Prefer bigger AND rounder blobs.
        score = area * circularity
        if score > best_score:
            best_score = score
            # The ball's area is reported as pi*r^2 from the fitted circle,
            # which is smoother than the raw contour area.
            best = {"center": (x, y), "radius": r, "area_px": math.pi * r * r,
                    "circularity": circularity}
    # best is None if no round yellow-green blob was found.
    return best, mask


# =============================================================================
# TRACKING: giving each blue object a stable ID across frames
# =============================================================================
class Track:
    # One Track = one blue object followed over time.
    def __init__(self, tid, obj, trail_len):
        self.id = tid                       # number shown as B1, B2, ...
        self.center = obj["center"]         # current centre (x, y)
        self.anchor = obj["center"]  # last point counted towards distance
        # Trail of past centres. maxlen means only the newest trail_len are kept.
        self.trail = deque([obj["center"]], maxlen=trail_len)
        self.dist_px = 0.0                  # total distance travelled in pixels
        self.dist_cm = 0.0                  # total distance travelled in cm
        self.missed = 0                     # frames in a row it wasn't found
        self.obj = obj                      # this frame's blob (or None if missed)


class CentroidTracker:
    """Greedy nearest-centroid matching so each blue object keeps its ID."""

    def __init__(self, max_dist=120, max_missed=15, trail_len=64, min_step=3.0):
        self.max_dist = max_dist        # a blob more than 120 px away can't be the same object
        self.max_missed = max_missed    # forget an object after 15 frames unseen
        self.trail_len = trail_len      # trail keeps the last 64 centres
        self.min_step = min_step  # ignore jitter smaller than this (px)
        self.tracks = []                # all objects currently being followed

    def reset(self):
        # Forget everything (used by the 'c' key and after a mouse click).
        self.tracks = []

    def update(self, objects, px_per_cm):
        # Step 1: list EVERY (track, new blob) pair with the distance between
        # them, sorted from closest to farthest.
        # Each entry is (distance, track index, object index).
        pairs = sorted(
            (math.dist(t.center, o["center"]), ti, oi)
            for ti, t in enumerate(self.tracks)
            for oi, o in enumerate(objects)
        )
        # Sets remember which tracks and blobs have already been paired.
        used_t, used_o = set(), set()
        # Step 2: "greedy" matching. Walk from the closest pair upwards and
        # accept a pair if neither side is taken yet and it is close enough.
        for d, ti, oi in pairs:
            if d > self.max_dist or ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            t, o = self.tracks[ti], objects[oi]
            # Move the track to its new position; it was seen, so missed = 0.
            t.center, t.obj, t.missed = o["center"], o, 0
            # How far has it moved since the last COUNTED point?
            step = math.dist(t.anchor, t.center)
            if step >= self.min_step:
                # Real movement (not camera jitter): add it to the totals.
                t.dist_px += step
                if px_per_cm:
                    # Only count cm once the scale is known.
                    t.dist_cm += step / px_per_cm
                # This point becomes the new reference and joins the trail.
                t.anchor = t.center
                t.trail.append(t.center)

        # Step 3: tracks with no matching blob this frame.
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
                t.obj = None    # nothing to draw for it this frame
        # Drop tracks that have been missing for too long.
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        # Step 4: blobs that matched no track are new objects -> new IDs.
        for oi, o in enumerate(objects):
            if oi not in used_o:
                # Reuse the lowest ID nobody is holding, so with 3 objects
                # on screen the labels stay B1-B3 instead of climbing forever.
                taken = {t.id for t in self.tracks}
                tid = next(i for i in itertools.count(1) if i not in taken)
                self.tracks.append(Track(tid, o, self.trail_len))

        # Return only the tracks visible in THIS frame (they have a blob).
        return [t for t in self.tracks if t.obj is not None]


# =============================================================================
# PIPELINE: ties all the steps together for one frame
# =============================================================================
class Pipeline:
    def __init__(self, marker_cm, min_blue_area, min_ball_area):
        self.scale = ScaleEstimator(marker_cm)
        self.tracker = CentroidTracker()
        # .copy() so changing the blue range later never changes DEFAULT_BLUE.
        # This line is a generator unpacked into two variables: lo and hi.
        self.blue_lo, self.blue_hi = (a.copy() for a in DEFAULT_BLUE)
        self.min_blue_area = min_blue_area
        self.min_ball_area = min_ball_area
        # The latest HSV frame, kept so a mouse click can read its colour.
        self.hsv = None

    def process(self, frame):
        # Blur with a 5x5 Gaussian (0 = let OpenCV pick the spread). Removes
        # pixel noise so colour masks come out cleaner.
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        # Convert BGR -> HSV so colour (hue) is separate from brightness.
        self.hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # ArUco detection needs a single-channel grey image (unblurred, so
        # marker edges stay sharp).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. The ball (found first, so the scale can fall back on its size).
        ball, ball_mask = find_ball(self.hsv, self.min_ball_area)
        # 2. Scale from the marker, or from the ball if no marker is visible.
        corners = self.scale.update(gray, ball)
        # 3. Blue objects (using the current, possibly clicked, blue range).
        blue, blue_mask = find_blue(self.hsv, self.blue_lo, self.blue_hi, self.min_blue_area)
        # 4. Give blue objects IDs, trails and distances.
        tracks = self.tracker.update(blue, self.scale.px_per_cm)
        # Hand everything back in one dictionary for the drawing functions.
        return {"tracks": tracks, "ball": ball, "corners": corners,
                "blue_mask": blue_mask, "ball_mask": ball_mask}

    def select_blue(self, x, y):
        """Centre the blue HSV range on the colour under the clicked pixel."""
        if self.hsv is None:
            return
        h_img, w_img = self.hsv.shape[:2]
        # Take a 7x7 patch around the click. max/min keep it inside the image.
        patch = self.hsv[max(y - 3, 0):min(y + 4, h_img), max(x - 3, 0):min(x + 4, w_img)]
        # Median H, S, V of the patch: robust to one or two odd pixels.
        h, s, v = np.median(patch.reshape(-1, 3), axis=0)
        # New range: hue +/-10 around the clicked hue, generous S and V.
        # max/min clamp values to legal HSV limits. uint8 = 0-255 integers,
        # the type OpenCV's inRange expects.
        self.blue_lo = np.array([max(h - 10, 0), max(s - 70, 50), max(v - 70, 40)], dtype=np.uint8)
        self.blue_hi = np.array([min(h + 10, 179), 255, 255], dtype=np.uint8)

    def reset_blue(self):
        # Go back to the default blue range ('r' key).
        self.blue_lo, self.blue_hi = (a.copy() for a in DEFAULT_BLUE)


# =============================================================================
# SMALL DRAWING HELPERS
# =============================================================================
def pt(p):
    # OpenCV drawing needs whole-number pixel coordinates; round the floats.
    return (int(round(p[0])), int(round(p[1])))


def relative(obj_center, ball, scale):
    """Blue object's offset from the ball (left/right, up/down) and distance."""
    # How far right (dx) and down (dy) the blue object is from the ball.
    dx = obj_center[0] - ball["center"][0]
    dy = obj_center[1] - ball["center"][1]
    # Straight-line distance: sqrt(dx^2 + dy^2).
    d = math.hypot(dx, dy)
    # Show the signs as words: negative dx = left of the ball, negative dy = above it.
    side = "left" if dx < 0 else "right"
    level = "up" if dy < 0 else "down"  # image y grows downwards
    if scale.px_per_cm:
        k = scale.px_per_cm
        # :.1f shows one decimal; abs() drops the sign since side/level say it.
        return f"{side} {abs(dx) / k:.1f} {level} {abs(dy) / k:.1f} d={d / k:.1f} cm"
    return f"{side} {abs(dx):.0f} {level} {abs(dy):.0f} d={d:.0f} px"


def put(img, text, org, color=WHITE, size=0.5, thickness=1):
    # Draw the text twice: first thick and black (an outline), then in colour
    # on top. This keeps labels readable on any background.
    # org = bottom-left corner of the text. LINE_AA = smooth (anti-aliased) edges.
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


# =============================================================================
# LEFT SIDE: drawings on top of the live camera image
# =============================================================================
def draw_camera(frame, result, scale):
    tracks, ball = result["tracks"], result["ball"]

    # Outline any ArUco marker found this frame.
    if result["corners"]:
        cv2.aruco.drawDetectedMarkers(frame, result["corners"])

    # Grey lines between every PAIR of blue objects, labelled with distance.
    # tracks[i + 1:] means "only the ones after a", so each pair is drawn once.
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            cv2.line(frame, pt(a.center), pt(b.center), GRAY, 1, cv2.LINE_AA)
            # Midpoint of the line, where the distance label goes.
            mid = ((a.center[0] + b.center[0]) / 2, (a.center[1] + b.center[1]) / 2)
            put(frame, scale.length(math.dist(a.center, b.center)), pt(mid), GRAY, 0.45)

    for t in tracks:
        c = t.obj["contour"]
        # The object's real outline. [c] because drawContours takes a list;
        # -1 = draw all contours in that list; 2 = line thickness.
        cv2.drawContours(frame, [c], -1, BLUE_OUTLINE, 2)
        # Bounding box: minAreaRect finds the smallest ROTATED rectangle
        # around the shape; boxPoints gives its 4 corners; drawContours draws it.
        box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.int32)
        cv2.drawContours(frame, [box], 0, BOX_COLOR, 1)
        # The trail: a line through the stored past centres.
        if len(t.trail) > 1:
            cv2.polylines(frame, [np.array([pt(p) for p in t.trail], np.int32)], False, BLUE_OUTLINE, 2)
        cx, cy = pt(t.center)
        # A filled dot at the centre (-1 thickness = filled).
        cv2.circle(frame, (cx, cy), 4, WHITE, -1)
        # Label: ID and area, e.g. "B1 15.3 cm^2".
        put(frame, f"B{t.id} {scale.area(t.obj['area_px'])}", (cx + 8, cy - 8))
        if ball:
            # Red line from the object to the ball, plus dx/dy/distance text.
            cv2.line(frame, (cx, cy), pt(ball["center"]), LINK_COLOR, 1, cv2.LINE_AA)
            put(frame, relative(t.center, ball, scale), (cx + 8, cy + 12), size=0.45)

    if ball:
        bc, r = pt(ball["center"]), int(round(ball["radius"]))
        # Circle around the ball and a dot at its centre.
        cv2.circle(frame, bc, r, BALL_COLOR, 2)
        cv2.circle(frame, bc, 4, BALL_COLOR, -1)
        label = f"Ball c=({bc[0]},{bc[1]}) r={r}px A={scale.area(ball['area_px'])}"
        # Measure the label's width so it can be kept inside the frame.
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Clamp x so the text never runs off the left or right edge.
        lx = min(max(bc[0] - r, 5), frame.shape[1] - tw - 5)
        # Put it just above the ball (but never above y = 45).
        put(frame, label, (lx, max(bc[1] - r - 10, 45)), BALL_COLOR)

    # Top-left status line about the scale.
    if scale.px_per_cm:
        # Say which reference the scale came from this frame.
        ref = {"marker": f"{scale.marker_cm} cm marker", "ball": f"{BALL_CM} cm ball",
               "held": "held, nothing visible"}[scale.source]
        hud = f"Scale: {scale.px_per_cm:.2f} px/cm ({ref})"
    else:
        hud = "Scale: no marker or ball yet - showing px"
    put(frame, hud, (10, 25), size=0.6)
    # Key hints along the bottom edge.
    put(frame, "click blue to select | c clear | r reset | m masks | q quit",
        (10, frame.shape[0] - 12), GRAY, size=0.45)
    return frame


# =============================================================================
# RIGHT SIDE: black text panel with the numbers
# =============================================================================
def draw_panel(h, result, scale, panel_w=380):
    """Pitch-black text panel shown to the right of the camera view."""
    # np.zeros makes an all-black image, as tall as the camera frame.
    # uint8 = normal 0-255 image pixels.
    panel = np.zeros((h, panel_w, 3), np.uint8)
    tracks, ball = result["tracks"], result["ball"]

    # x0 = left edge of the panel text; y = current line's height (moves down).
    x0, y = 12, 28
    # A grey divider between the camera view and the panel.
    cv2.line(panel, (0, 0), (0, h), GRAY, 1)
    put(panel, f"Blue objects: {len(tracks)}", (x0, y), size=0.6)
    y += 26
    if ball:
        bc = pt(ball["center"])
        put(panel, f"Ball: ({bc[0]}, {bc[1]}) r={ball['radius']:.0f}px", (x0, y))
        y += 20
        put(panel, f"      A={scale.area(ball['area_px'])}", (x0, y))
    else:
        put(panel, "Ball: not detected", (x0, y), GRAY)
    y += 30

    # One block of lines per blue object.
    for t in tracks:
        if y > h - 20:
            # Out of room at the bottom of the window: stop writing.
            break
        cx, cy = pt(t.center)
        # Distance travelled in cm if we have a scale, otherwise in px.
        traversed = f"{t.dist_cm:.1f} cm" if scale.px_per_cm else f"{t.dist_px:.0f} px"
        put(panel, f"B{t.id}: (x={cx}, y={cy})  A={scale.area(t.obj['area_px'])}", (x0, y))
        y += 20
        if ball:
            put(panel, f"   wrt ball: {relative(t.center, ball, scale)}", (x0, y), size=0.45)
            y += 20
        put(panel, f"   traversed: {traversed}", (x0, y), size=0.45)
        y += 26
    return panel


# =============================================================================
# MAIN: startup, then the loop that runs every frame
# =============================================================================
def main():
    # Open the camera (or video file) and ask for FRAME_W x FRAME_H frames.
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        # SystemExit stops the program with this message.
        # !r prints the value with quotes if it is a string (e.g. a file name).
        raise SystemExit(f"Could not open video source {SOURCE!r}")

    # Build the pipeline (scale estimator + tracker inside it).
    pipe = Pipeline(MARKER_CM, MIN_BLUE_AREA, MIN_BALL_AREA)
    show_masks = False

    # Create the window up front so the mouse callback can attach.
    cv2.namedWindow("Tracker")
    # Width of the camera part of the window; clicks right of it are on the
    # text panel and are ignored. Set once the first frame arrives.
    frame_w = 0

    # OpenCV calls this function on every mouse event in the window.
    # *_ swallows the extra arguments (flags, param) we don't use.
    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and x < frame_w:
            # Left click: retune the blue range to the clicked colour and
            # restart tracking with the new range.
            pipe.select_blue(x, y)
            pipe.tracker.reset()

    cv2.setMouseCallback("Tracker", on_mouse)

    # ---------------- The main loop: once per frame ----------------
    while True:
        # Grab one frame. ok is False if the camera stops or the video ends.
        ok, frame = cap.read()
        if not ok:
            break
        # Mirror first, so everything after (detection, left/right, clicks) uses the mirrored view.
        if MIRROR:
            frame = cv2.flip(frame, 1)
        # Do all the detection and tracking for this frame.
        result = pipe.process(frame)
        frame_w = frame.shape[1]
        # Draw on a COPY of the frame so the original stays untouched.
        view = draw_camera(frame.copy(), result, pipe.scale)
        # np.hstack glues the camera view and the text panel side by side
        # (both are the same height) so they show as one window.
        cv2.imshow("Tracker", np.hstack([view, draw_panel(frame.shape[0], result, pipe.scale)]))
        if show_masks:
            cv2.imshow("Blue mask", result["blue_mask"])
            cv2.imshow("Ball mask", result["ball_mask"])

        # waitKey(1) waits 1 ms for a key AND lets OpenCV refresh the windows.
        # Without it, the windows would never update. & 0xFF keeps only the
        # last 8 bits so the key code is the same on every operating system.
        key = cv2.waitKey(1) & 0xFF
        # ord("q") = the number for the letter q. 27 = the Esc key.
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            pipe.tracker.reset()
        elif key == ord("r"):
            pipe.reset_blue()
            pipe.tracker.reset()
        elif key == ord("m"):
            # Flip True <-> False.
            show_masks = not show_masks
            if not show_masks:
                cv2.destroyWindow("Blue mask")
                cv2.destroyWindow("Ball mask")

    # Clean up: release the camera and close every window.
    cap.release()
    cv2.destroyAllWindows()


# Run main() only when this file is run directly
# (python tracker_explained.py), not when it is imported by another file.
if __name__ == "__main__":
    main()
