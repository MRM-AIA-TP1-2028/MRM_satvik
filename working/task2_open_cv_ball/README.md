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
   - Ball: H 25–45, S 70–255, V 80–255

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

The test clip ([US Open rally](https://youtu.be/-jtV7IJP8NU), 720p, 2687 frames) is saved as `videos/test_video.mp4`. Each version writes a copy of the video with the tracking drawn on top.

| Version | Folder | Ball tracked |
|---|---|---|
| v1 `--mode original` | `v1/` | No real ball found (the hits are in the crowd) |
| v1 `--mode broadcast` | `v1/` | 1416 frames (52.7%) |
| v2 | `v2/` | 1980 frames (73.7%): 1637 detected + 343 interpolated, but **too many false positives** |
| v3 | `v3/` | 1552 frames (57.8%): 1408 detected + 144 interpolated, with fewer false positives than v1 |
| v4 | `v4/` | 1693 frames (63.0%): 1607 detected (261 from the guided search) + 86 interpolated, keeping all but 3 of v3's detected frames |
| v5 (TrackNet CNN) | `v5_tracknet/` | **2053 frames (76.4%)**: 1912 detected + 141 interpolated, with long continuous rally paths and few stray points |

The percentages count every frame, including the time between points and the end card, when no ball is in play.

### v1: frame-by-frame detection

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

### v2: offline tracking over the whole clip

```bash
cd v2
python track_ball_v2.py              # reuses cached candidates if present
python track_ball_v2.py --no-cache   # recompute candidates
```

v2 works on a saved video, so it can look at the whole clip before deciding where the ball is:
1. **Candidates.** The same cues as v1 with looser thresholds (motion 18 instead of 25, top-hat 22 instead of 30). Blobs near players are flagged rather than dropped, because the ball is next to a player at every hit.
2. **Tracklets.** Many hypotheses are followed at once with a constant-velocity model, so a false track can no longer take over from the real ball.
3. **Filtering.** A tracklet is kept only if it moves like a ball:
   - at least 5 points
   - at least 25 px of travel
   - a median speed of at least 3 px/frame (slower movers are people)
   - smooth motion, with median acceleration at most 6 px/frame²
   - not mostly next to a player
4. **Selection.** Weighted interval scheduling picks the best set of tracklets that don't overlap in time.
5. **Gap filling.** Two neighbouring tracklets are joined by linear interpolation only when they belong to one ball path:
   - For gaps of 3 frames or fewer (a hit or bounce), the distance must be reachable at the ball's speed.
   - For longer gaps (up to 12 frames), one tracklet's straight-line flight must land on the other.

   Interpolated positions are drawn as a grey ring labelled "interpolated" and counted separately.

`v2/output/v1_vs_v2_trajectories.jpg` plots every tracked position over 150-frame windows, with v1 and v2 side by side. Yellow is detected and magenta is interpolated. v2 gives much more complete arcs. Some errors remain:
- A few short false tracklets, for example along the centre service line and below the baseline.
- Gap filling is linear, so it cuts corners across a bounce.

v2 was rejected: it tracks more frames, but the extra coverage comes with too many false positives.

### v3: v1 plus conservative clean-up

```bash
cd v3
python track_ball_v3.py              # reuses cached v1 detections if present
python track_ball_v3.py --no-cache   # rerun v1's detector
```

v3 keeps v1's detector exactly as it is and loosens nothing. It then makes three conservative edits:
1. **Extend runs.** v1 never draws the two frames that confirm a new track, and it stops at the end of a run. Each run is extended by up to 8 frames at each end, but only onto v1's own candidates that sit within 6–12 px of the run's straight-line path. This adds 119 frames.
2. **Drop blips.** Runs of 1–2 detections that don't line up with a neighbouring run are removed. These were v1's scattered false hits near the umpire's chair, at players' feet and along the baseline. This removes 127 frames.
3. **Fill short gaps.** Gaps of up to 5 frames are interpolated, and only when one run's straight-line flight lands on the next run. This adds 144 frames, drawn as a grey ring labelled "interpolated".

`v3/output/v1_vs_v3_trajectories.jpg` compares v1 and v3. Yellow is detected, magenta is interpolated, and red crosses are v1 points that v3 dropped. A few of the dropped points were on real ball flights, so the clean-up trades a little recall for precision.

### Why v1 misses frames that look the same

I took 79 missed frames where the tracks on either side show roughly where the ball must be, and checked each stage of the detector at that spot. Zoomed examples are in `v3/output/missed_ball_examples.jpg`.

| Why the ball was missed | Share |
|---|---|
| No motion detected | 52% |
| Next to a player (excluded on purpose) | 20% |
| Detected, but the tracker rejected it | 13% |
| Motion and brightness don't overlap | 9% |
| Other | 6% |

The zoomed frames show two very different situations:
- **The ball isn't in the image.** At about 17 px/frame, motion blur and compression erase a 4-px ball. It also disappears when it crosses a white line or passes behind a player's feet. No detector can find it in these frames, and only interpolation can fill them.
- **The ball is visible but v1 rejects it.** The blob is only 2–3 px, so the 2×2 noise-removal step deletes it. Or its motion is just under the threshold. Or it is next to a player at a hit.

### v4: guided search where the ball is expected

```bash
cd v4
python track_ball_v4.py              # reuses v3's cached v1 detections
python track_ball_v4.py --no-cache   # rerun v1's detector
```

v1's strict rules are needed across the whole frame, otherwise crowd, feet and line shimmer get in. Right where a track says the ball must be, though, a false blob is unlikely. So v4 relaxes the rules only there:
1. **Strict pass.** v1's detector runs unchanged, so v4 starts from exactly the same detections as v1 and v3.
2. **Guided pass.** Each v1 run of 3 or more frames is followed for up to 6 frames forwards from its end and backwards from its start, along a curve fitted to the run (quadratic, so lobs are followed). At each predicted position, a small window (8 px plus 3 px per frame) is searched with relaxed rules:
   - no 2×2 noise removal
   - motion threshold 14 instead of 25
   - top-hat threshold 20 instead of 30
   - players allowed
   - still inside the play area
3. **v3's clean-up.** Extend runs, drop blips, fill short gaps.

The guided pass never changes v1's runs. It can only add points right next to existing tracks.

An earlier attempt ran the guided search inside the live tracker. That changed the tracker's state, so v4 lost tracks that v3 had and picked up new false paths at the edges. Doing the guided search afterwards avoids both problems.

`v4/output/v3_vs_v4_trajectories.jpg` compares v3 and v4. Cyan marks points added by the guided search. They mostly continue existing arcs: into a racket at a hit, out from the far player, and at the ends of flights. A few small clusters are doubtful, for example near the umpire's chair. Frames where the ball isn't in the image are still not recovered.

Even so, v4 still lost the ball at every hit and left a stale trail on screen. Watching the video, it didn't feel much better than v1: about 4 rule-based versions is the limit of hand-tuned rules on this footage.

### v5: TrackNet, a CNN built for this problem (recommended)

```bash
cd v5_tracknet
pip install gdown                                                  # once
gdown 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl -O weights/tracknet_tennis.pt   # once, 43 MB
python track_ball_tracknet.py              # reuses cached detections if present
python track_ball_tracknet.py --no-cache   # rerun the network (about 12 min on an M-series Mac GPU)
```

[TrackNet](https://arxiv.org/abs/1907.03698) (Huang et al., 2019) is a small encoder-decoder CNN for tracking tennis and badminton balls in broadcast video.
- **Input:** 3 consecutive frames stacked together (9 channels at 640×360).
- **Output:** a heatmap of where the ball is.

Because it sees 3 frames at once, it learns what a faint, blurred, 4-px ball looks like from its motion and surroundings. It no longer depends on thresholds that either miss the ball or pick up feet and lines. The architecture and pretrained tennis weights come from [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet). That repository shows no license, so the weights are not committed (`v5_tracknet/.gitignore`) and are downloaded instead. They are loaded with `torch.load(weights_only=True)`, which reads only tensors and never runs code from the file.

Post-processing:
1. Take the strongest blob in each heatmap.
2. Drop detections that jump more than 100 px from both neighbours.
3. Apply v3's clean-up: drop blips and fill short gaps that line up.

The trail is also drawn only while the ball is actually tracked, so a lost ball no longer leaves a frozen line.

**Results:**
- **More detections:** TrackNet detects the ball in 1912 frames (v1: 1416), for 76.4% coverage.
- **Whole rallies:** `v5_tracknet/output/v1_vs_v5_trajectories.jpg` shows long continuous paths for whole rallies, with the ball out, bouncing and back, and few of v1's stray points.
- **Ball found again after a hit:** `v5_tracknet/output/after_hit_sequence.jpg` shows the ball picked up again right after a hit and followed back across the court. Every rule-based version lost the ball at that point.

Frames where the ball is truly invisible (motion blur, crossing a white line) are still missed.

**Possible next steps**
- Fit parabolic arcs, piecewise between bounces, instead of straight lines.
- Use a Kalman smoother.
- Fine-tune TrackNet on a few hundred labelled frames from this kind of footage.
- Use TrackNetV3's background estimation and trajectory rectification for the remaining gaps.
