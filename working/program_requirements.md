# MRM Taskphase: Program requirements

Everything your programs must do, taken from the task posts and meeting minutes in the AIA Taskphase 2028 chat. Newest task first. As of 21 Oct 2026.

## 1. Blue object + tennis ball with scale estimation (current)

Posted 20 Oct. **Deadline: 23 Oct 2026 (Fri).**

**Input**
- [ ] Reads a live camera stream

**Detection and tracking**
- [ ] Detects a tennis ball and blue objects in the same stream
- [ ] Lets the user select blue objects, then identifies and tracks them while still detecting the ball
- [ ] Treats the blue object and the tennis ball as separate objects
- [ ] Does not assume the blue object is circular (any shape)
- [ ] Detects multiple blue objects (19 Oct meeting)

**Camera window must show**
- [ ] Blue object: area in **cm²**
- [ ] Blue object: bounding outline drawn with `cv2.drawContours()`, not `cv2.boundingRect()` (19 Oct meeting)
- [ ] Blue object: a line tracking its centre
- [ ] Blue object: its position relative to the tennis ball, when the ball is visible
- [ ] Tennis ball: centre
- [ ] Tennis ball: radius in **px**
- [ ] Tennis ball: area in **cm²**

**Separate info window (19 Oct meeting)**
- [ ] Separate window with a pitch-black background
- [ ] Each blue object's (x, y) centre coordinates
- [ ] Each blue object's position relative to the yellow ball
- [ ] Distance from each blue object to the other objects
- [ ] Distance travelled by each blue object, tracked separately

**Scale**
- [ ] Converts px to cm. ArUco markers may be used as the scale reference.

**Submission**
- [ ] Code pushed to your GitHub repository
- [ ] README.md covering: your approach, the method you used, and other possible methods

## 2. Blue object + tennis ball detection, px version (done)

Posted 13 Oct, deadline 17 Oct. The same as task 1, but with areas in px² instead of cm²:
- Blue object: area in px², bounding box, centre-tracking line and position relative to the ball
- Tennis ball: centre, radius in px and area in px²
- Same GitHub + README submission

Task 1 replaces this version, so meeting task 1's requirements covers both.

## 3. ArUco marker pose estimation (done)

Posted 7 Oct, deadline extended to 12 Oct.
- [ ] Python + OpenCV script using the laptop webcam in real time
- [ ] Detects ArUco markers in the 4×4, 5×5 and 6×6 dictionaries
- [ ] Estimates each marker's pose (orientation and position) and depth
- [ ] Shows the video stream with the detected markers and their pose overlaid
- [ ] Pushed to the MRM GitHub organisation before the 12 Oct meeting

## Things reviewers asked about (be ready to explain in your code)

- Camera matrix and distortion coefficients: what they are and why they're zero when uncalibrated
- Why HSV instead of BGR
- `cv2.moments()`, Gaussian blur vs `morphologyEx()`, Hough circles vs circularity
- `cv2.Rodrigues()` and the main `cv2.aruco` functions
- `if __name__ == "__main__":`

Answers are in `MRM_satvik/review questions/Review Questions.docx`.
