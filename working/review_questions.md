# MRM Review Questions

Six meetings produced 42 review questions, listed under the meeting that set them. Your Review Questions doc (7 Oct) answers the 17 Sep and 5 Oct sets. The 14 Sep, 7 Oct, 9 Oct and 19 Oct sets are still open.

## 19 Oct meeting (open, latest)

1. How many distortion coefficients are there and what are they?
2. What is the difference between .npz and .yaml files?
3. What does `cv2.moments()` do? Explain all its attributes.
4. What is the difference between Gaussian blur and `morphologyEx()`? Explain both in depth.
5. In addition to the current code, create a separate black-and-white window that shows each blue object's centre (x, y), its position relative to the yellow tennis ball and its distance to the other objects.
6. What is the difference between original ArUco and 5×5 ArUco, and why do we need original ArUco?
7. What functions are there in `cv2.aruco`?
8. What is the fundamental difference between Hough circles and circularity? Which is better, and why?
9. What is the difference between methods and functions?
10. What is `cv2.Rodrigues()`?
11. Why use HSV and not BGR?

## 9 Oct meeting (open)

1. What is the purpose of `camera_matrix` in the code? Is the camera matrix the same for all devices?
2. Why is the orientation coefficient initialised to 0 in the code?
3. What is the purpose of `if __name__ == "__main__":` in Python? Does the code in this block still run if `__main__` is changed to another name?

## 7 Oct meeting (open)

1. What is /etc/netplan? (answered in the 5 Oct set)
2. What are point releases (Ubuntu sub-versions)?
3. What is the difference between Ubuntu 24.04 and 24.04.5?
4. Write Python code that reverses a word using string slicing.
5. What does `.strip()` do?
6. What is the difference between an array and a list?

## 5 Oct meeting (answered)

1. What is an IMU?
2. What is yaw, and how does a 9 DOF IMU reduce it?
3. What is /etc/netplan?
4. What does /etc/netplan do in a Linux distro?
5. Is a cloud service a virtual machine? Are all cloud services VMs? Differentiate between them.
6. What are the advantages of Docker?

## 17 Sep meeting (answered)

1. What is the difference between a 6 DOF and a 9 DOF IMU?
2. What are the 3 degrees of freedom of the IMU's magnetic component?
3. Encoder vs IMU: which is better for the arm?
4. Why is a stereo camera digital?
5. Where would you use an analog camera vs a digital camera?
6. git reset vs git revert?
7. git pull vs git fetch?
8. How many levels are there in git config?

At this meeting you were also asked to add a review-questions section to the report and to state that all of Report 2 is a review.

## 14 Sep meeting (mostly open)

1. What is an IMU? (answered)
2. How many degrees of freedom does an IMU provide? (partly answered under 6 vs 9 DOF)
3. What is a wheel encoder, and where is it used?
4. What is a stereo camera, and how is depth perceived from it?
5. What is a digital camera, and how does it differ from an analog camera? (partly answered)
6. What is LiDAR?
7. What is the difference between 2D and 3D LiDAR?
8. Why would a rover use both LiDAR and a stereo camera?

Sources: the AIA Taskphase 2028 WhatsApp export in `mrm_Repo/working/`, and `review questions/Review Questions.docx`.
