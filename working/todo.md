# MRM Taskphase: To-do list

From the AIA Taskphase 2028 WhatsApp chat (24 Aug to 20 Oct 2026). As of 21 Oct 2026.

Next deadline: the scale-estimation task, due **Fri 23 Oct**. Compulsory meeting **Wed 21 Oct, 5 PM**, MRM Workshop (Bay 9).

## Due now

- [ ] Go to the Taskphase meeting (21 Oct, 5 PM, MRM Workshop Bay 9). If you'll be absent or late, post Name / Registration Number / Reason in the group beforehand.
- [ ] Make the code changes asked for at the 19 Oct meeting:
    - [ ] Detect multiple blue objects
    - [ ] Add a separate pitch-black window showing each blue object's (x, y), its position relative to the yellow ball, and the distance each object has travelled
    - [ ] Draw outlines with `cv2.drawContours()` instead of `cv2.boundingRect()`
- [ ] Answer the 11 review questions from the 19 Oct meeting (see review_questions.md). Q5 is a code task.
- [ ] Scale-estimation task: report the blue object's area in cm² and the ball's centre, radius in px and area in cm², with ArUco markers for scale. Push it to GitHub with a README covering your approach, method and other possible methods. **Deadline: 23 Oct 2026.**

`tracker.py` and `README.md` in the repo already seem to cover the 19 Oct changes and the cm² scale, but they aren't committed yet. What's left is to test them on the webcam, commit and push.

## Older review questions still unanswered

Your Review Questions doc (7 Oct) covers only the 17 Sep and 5 Oct sets.

- [ ] 14 Sep set: wheel encoder, stereo depth, LiDAR, 2D vs 3D LiDAR, LiDAR plus stereo
- [ ] 7 Oct set: Ubuntu point releases, string slicing, `.strip()`, array vs list
- [ ] 9 Oct set: `camera_matrix`, orientation coefficient, `if __name__ == "__main__"`

Review answers go at the end of your next report, or in the review-questions form when one is shared.

## Past tasks: check each was submitted

| Task | Deadline | How to submit |
| --- | --- | --- |
| Blue object + tennis ball detection (px) | 17 Oct | GitHub + README |
| ArUco pose estimation (4×4, 5×5, 6×6) | 12 Oct (extended) | Push to GitHub org |
| GitHub org form + account with the same email | 11 Oct | Form |
| Install Ubuntu 24.04 LTS (VM on Apple-silicon Mac) | 12 Oct | Show at meeting |
| Python course (YouTube) | 7 Oct | — |
| Review questions form (5 Oct set) | 7 Oct, 4:59 PM | Form |
| Report 3: Linux Filesystem | 3 Oct | Form |
| Report 2: Version Control (state that the whole report is a review) | 16 Sep | Form |
| Report 1: Rovers | 13 Sep | Form |

## Standing rules

- Reports follow the template and its naming format.
- Reports and images must not be AI-generated, and every source is cited at the end.
- Meetings are compulsory. Absences use the Name / Reg No / Reason format, posted in advance.
