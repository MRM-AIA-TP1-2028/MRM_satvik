# Task 3 - Blue objects + tennis ball, measured in cm

This one uses the laptop camera to find blue objects and a tennis ball, track
them as they move, and give their sizes and distances in actual cm instead of
pixels.

The trick for getting real units is an ArUco marker. You print it, measure it
with a ruler, and since the program knows how big it really is, it can work out
how many pixels make a cm. Everything else gets converted using that.

How it works, roughly:
- blue stuff is picked out by colour (HSV), any shape works
- the tennis ball is found by its yellow-green colour plus how round it is
- each blue object keeps an ID and leaves a trail so you can see where it went
- a second black window lists positions, distance from the ball, and how far
  each object has moved

To run it:

    pip install -r requirements.txt
    python make_marker.py
    python tracker.py

Print marker.png, measure the black square and put that number in MARKER_CM at
the top of tracker.py.

Keys: click a blue object to lock onto its shade, c clears trails, r resets the
colour, m shows the masks, q quits.

Couple of things to know: the cm numbers are only right for stuff that's about
as far from the camera as the marker is. Lighting also messes with the colours
a bit. If you're on a Mac, turn off Center Stage because it zooms around and
throws off the scale.
