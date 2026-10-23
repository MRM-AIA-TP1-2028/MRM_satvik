# Blue Object Selection and Tennis Ball Detection with Scale Estimation

A Python + OpenCV program that reads a live camera stream, detects and tracks
**blue objects** (of any shape) and a **tennis ball** as separate objects, and
reports sizes in **cm** by using an ArUco marker as the scale reference.

## Running it

```bash
pip install -r requirements.txt
python make_marker.py     # writes marker.png: print it, measure the black square
python tracker.py
```

Settings are constants at the top of `tracker.py`. Edit them before running:

| Setting | Default | Meaning |
| --- | --- | --- |
| `SOURCE` | `0` | camera index (0 = MacBook Air camera), or a video file path |
| `MARKER_CM` | `5.0` | side length of the marker's black square, in cm (set this to your measurement) |
| `MIN_BLUE_AREA` | `600` | ignore blue blobs smaller than this (px²) |
| `MIN_BALL_AREA` | `300` | ignore ball candidates smaller than this (px²) |
| `FRAME_W`, `FRAME_H` | `1280`, `720` | camera resolution requested (the MacBook camera can do 1080p, but 720p fits the screen and runs faster) |

On a MacBook, turn **Center Stage off** (Control Centre → Video Effects) while measuring. It zooms and pans the image, which changes the px-per-cm scale.

Keys: **click** a blue object to select its colour · `c` clear trails ·
`r` reset colour · `m` show HSV masks · `q` quit.

## What it shows

**Camera window**
- Blue objects: the outline (`cv2.drawContours`), a rotated bounding box,
  area in cm², a trail tracking the center, and the position relative to the
  tennis ball (dx, dy, distance in cm) with a line to the ball.
- Tennis ball: center, radius in px, and area in cm².
- The current scale (px/cm) and whether the marker is visible or the last value is held.

**Info window (pitch-black background)**
- A white map of every blue object's center, the ball, and the lines between them,
  plus the distance between each pair of blue objects.
- For each blue object: (x, y) in px, area, position relative to the ball, and
  the distance it has traveled so far.

Offsets use image axes: **+x is right, +y is down**. Until a marker has been seen,
every value is shown in px.

## Approach

1. **Pre-processing.** A Gaussian blur (5×5) removes sensor noise, then the
   frame is converted from BGR to **HSV**. Hue separates colour from lighting,
   so a single hue band holds up under brightness changes far better than BGR
   thresholds.
2. **Blue objects.** `cv2.inRange` on a blue hue band (H 100–130) builds a mask.
   A morphological open removes speckles and a close fills holes. Each external
   contour above the minimum area counts as one blue object, with no shape
   assumption. The center comes from image moments (`m10/m00`, `m01/m00`), and the
   area from `cv2.contourArea`. Clicking an object re-centres the hue band on that
   pixel's colour ("selection"), which helps with different shades of blue.
3. **Tennis ball.** The same process runs on a yellow-green band (H 25–45). Each
   candidate is scored on **circularity** (4πA/P²) and **fill ratio** (contour
   area / enclosing-circle area). The best candidate above both thresholds is the
   ball. `cv2.minEnclosingCircle` gives the center and radius, and the area is πr².
4. **Scale estimation.** Each frame is searched for a 4×4 ArUco marker
   (`DICT_4X4_50`, IDs 0–49; `make_marker.py` prints ID 0). The mean side length in px divided by the real side length in cm
   gives **px/cm**, smoothed with an exponential moving average and held if the
   marker drops out. Areas are divided by (px/cm)², and lengths by px/cm.
5. **Tracking.** A greedy nearest-centroid tracker matches each blue object to
   its track from the last frame, so IDs and trails stay stable. Jitter under
   3 px is ignored when adding up the distance traveled.

## Limitations

- The scale is only correct for objects at roughly the **same distance from the
  camera as the marker**, in a plane facing the camera. Objects nearer to the
  camera than the marker read too large.
- The colour thresholds depend on lighting. Use click-to-select or press `m`
  to check the masks.

## Other possible methods

- **Scale:** use full pose estimation (`cv2.aruco.estimatePoseSingleMarkers` /
  `solvePnP` with a calibrated camera matrix) to get each object's depth and
  correct the scale per object. Alternatively, use the tennis ball itself as the
  reference, since its diameter is a known ~6.7 cm.
- **Ball detection:** `cv2.HoughCircles` on a grayscale or masked image. This is
  more robust to partial occlusion, but slower and has more parameters to tune.
- **Colour segmentation:** k-means clustering, or adaptive thresholds learned from
  a selected region (histogram back-projection with `cv2.calcBackProject`).
- **Tracking:** OpenCV trackers (CSRT/KCF), or a Kalman filter per object to
  smooth trails and keep IDs through brief occlusions.
- **Learning-based detection:** a small YOLO model trained on tennis balls and blue objects.
