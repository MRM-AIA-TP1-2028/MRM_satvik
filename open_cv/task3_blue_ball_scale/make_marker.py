"""Generate a printable ArUco marker for scale estimation.

    python make_marker.py

Writes marker.png (DICT_4X4_50, ID 0), the only dictionary tracker.py searches. Print it, measure the black
square's side with a ruler, and set MARKER_CM in tracker.py to that length.
"""

import cv2

from tracker import ARUCO_DICT

MARKER_ID = 0
SIZE_PX = 600   # marker size in pixels
OUT = "marker.png"


def main():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, SIZE_PX)
    border = SIZE_PX // 6  # white quiet zone so the detector can find the edge
    marker = cv2.copyMakeBorder(marker, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(OUT, marker)
    print(f"Wrote {OUT} (4x4, id {MARKER_ID})")


if __name__ == "__main__":
    main()
