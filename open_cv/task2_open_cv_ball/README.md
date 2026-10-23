# Blue Object Selection and Tennis Ball Detection (OpenCV)

`ball_and_blue.py` reads a live camera stream. It finds and tracks blue objects of any shape and detects a tennis ball at the same time. The two are separate objects with separate detectors.

## Running

```bash
pip install opencv-python numpy        # or: conda activate SANDBOX-MAX
python ball_and_blue.py                # default webcam
python ball_and_blue.py --source 1     # another camera index
python ball_and_blue.py --source clip.mp4
```

| Input | Action |
|---|---|
| Left-click on a blue object | Select it, so only that object is tracked and annotated |
| Right-click | Clear the selection and track every blue object |
| `t` | Open HSV sliders to tune both colour ranges live |
| `m` | Show the binary masks (blue \| ball) |
| `c` | Clear the centre trails |
| `s` | Save a screenshot |
| `q` / `Esc` | Quit |

## What is shown on screen

**Blue object** (red when selected, light blue otherwise)
- Area in px² (the area inside the contour, so it follows the real shape)
- Bounding box, with its width × height
- A fading trail line through the last 64 centre positions
- Position relative to the tennis ball, when a ball is visible: a magenta line from the ball's centre to the object's centre, plus `dx`, `dy`, a plain-language direction ("120px left, 40px above"), the straight-line distance and the angle

**Tennis ball** (yellow)
- Centre `(x, y)`
- Radius in px
- Area in px² (πr²)

## Approach

The two targets differ mainly in colour, so I split the problem into two independent colour segmentations:

1. **One mask per object type.** Blue and optic yellow-green sit far apart on the HSV hue axis, so a blue pixel can never count as ball and the reverse. This keeps them separate objects, as the task requires.
2. **Shape is only used where it is known.** A tennis ball is always round, so its detector checks how circular each blob is. The blue object can be any shape, so its detector accepts any contour and measures it with contour area, image moments (for the centre) and a bounding rectangle. It never fits a circle.
3. **Selection needs stable identity.** To let the user click one blue object and keep following it, each blob needs an ID that survives from frame to frame. A small centroid tracker provides this, and the same IDs also give every object its own trail.

## Method

For each frame:

1. **Pre-processing.** Mirror the image (webcam only), apply an 11×11 Gaussian blur to reduce sensor noise, then convert BGR → HSV. HSV keeps colour (H) apart from brightness (V), so thresholds hold up better under changing light than they would in RGB.
2. **Colour thresholding** with `cv2.inRange`:
   - Blue: H 100–130, S 120–255, V 50–255
   - Ball: H 22–43, S 64–255, V 127–255

   Both ranges can be changed live with the `t` sliders.
3. **Morphology.** An opening removes speckles and a closing fills small holes (5×5 elliptical kernel, 2 iterations each).
4. **Contours** come from `cv2.findContours` (external contours only).
5. **Blue objects.** Keep every contour with area ≥ 800 px². For each one:
   - area = `cv2.contourArea`
   - centre = centroid from `cv2.moments`
   - box = `cv2.boundingRect`
6. **Tennis ball.** For each contour, fit `cv2.minEnclosingCircle` and compute circularity = contour area / (πr²). Reject blobs with r < 8 px or circularity < 0.65, which removes yellow objects that are not round. The largest blob that passes is the ball. Its centre, radius and area (πr²) come from the fitted circle.
7. **Tracking.** Pair each existing track with the nearest new centre, greedily, as long as the centre moved less than 90 px. Detections left unmatched start new IDs. A track that goes missing for more than 15 frames is dropped. Each track keeps a `deque` of its recent centres, which is drawn as the trail with thicker lines for newer points.
8. **Selection.** A left-click runs `cv2.pointPolygonTest` against each visible blue contour, and the object containing the click becomes the selected ID. If that ID is lost, the program goes back to tracking all blue objects.
9. **Relative position.** Compute `dx = x_blue − x_ball` and `dy = y_blue − y_ball`, then the distance `hypot(dx, dy)` and the angle `atan2(−dy, dx)`. The minus sign is there because image y points down, so after flipping it 0° means right and 90° means up.

## Other possible methods

- **Hough Circle Transform** (`cv2.HoughCircles`) on a grayscale or masked image to find the ball. It is robust to partial colour changes but slower and sensitive to its parameters. A good combination is to use it to confirm the colour-mask candidate.
- **Click-to-sample colour.** Read the HSV value under the clicked pixel and build the blue range around it (±10 H, for example), instead of using a fixed range.
- **OpenCV object trackers** (CSRT, KCF, MOSSE via `cv2.TrackerCSRT_create`). Select the blue object once with `cv2.selectROI` and let the tracker follow it. This handles brief colour dropouts but drifts over time and does not re-detect.
- **Back-projection + CamShift/MeanShift.** Build a hue histogram of the selected region and track it with `cv2.CamShift`. This also gives a rotated box for objects that are not circular.
- **Kalman filter** on each track's centre, to smooth the trail and predict the position through short occlusions.
- **Deep-learning detectors** such as YOLO (COCO already has a "sports ball" class) or a model fine-tuned on tennis balls. These are far more robust to lighting and background clutter but need a GPU or a lighter model to run in real time. Blue objects would still need colour filtering or a custom-trained class.
- **Adaptive lighting handling.** Apply CLAHE to the V channel, or work in the LAB colour space, to keep the thresholds stable as lighting changes.

## Testing the ball detector on a broadcast video

`v1/test_ball_on_video.py` runs the ball detector on a saved video and writes a copy with the tracking drawn on top: circle, centre, radius, area and a trail.

The test clip is a [US Open rally](https://youtu.be/-jtV7IJP8NU) (720p, 2687 frames), saved as `videos/test_video.mp4`. The tracked results are in `v1/output/`:
- `test_video_tracked_broadcast.mp4`: broadcast mode
- `test_video_tracked_original.mp4`: the original detector

| Mode | Ball tracked |
|---|---|
| `--mode original` | No real ball found (the hits are in the crowd) |
| `--mode broadcast` | 1416 of 2687 frames (52.7%) |

The percentage counts every frame, including the time between points and the end card, when no ball is in play.

```bash
cd v1
python test_ball_on_video.py --mode original    # the detector above, unchanged
python test_ball_on_video.py --mode broadcast   # adapted for TV footage (default)
```

**The original detector does not work on this footage.** Its hits are yellow blobs in the crowd, never the ball. The reasons:
- At 720p the ball is only about 4 px across.
- Video compression stores colour at lower resolution than brightness, so the ball's yellow is averaged into the court around it. Measured on the clip, the ball's hue is about 107, which is **blue**.

No HSV yellow range can find it.

**Broadcast mode** keeps the same idea (segment, then filter blobs by shape) but uses the cues that survive in broadcast video:
1. **Play-area mask.** The blue court plus the green surround, taken as a convex hull. This excludes the crowd and also non-court shots.
2. **Motion by double frame differencing.** A pixel counts as moving only if it differs from both of the two previous frames. Unlike background subtraction, this ignores the 1-px shimmer of court lines when the camera pans.
3. **Brightness.** A top-hat filter finds spots brighter than their surroundings and low in saturation. The yellow-green mask is also kept, for frames where the ball is big enough to show its colour.
4. **Size and shape.** Small, roughly round blobs only. Blobs on or near players are rejected.
5. **Constant-velocity tracking.** A track starts only after three detections in a row at a steady velocity.

**What limits v1:** its tracking is fragmented. Of its 298 tracked runs, 263 are shorter than 10 frames, and 291 of the gaps between runs last 10 frames or less. When the ball drops out briefly, or changes direction at a hit or bounce, the single track dies, and it needs three fresh frames to restart.
