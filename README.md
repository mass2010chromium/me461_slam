# CS 445 / ME 461 fall 2021 final project

## Single camera + IMU/encoders indoor exploration and simple SLAM
also has yolo object detector since why not?

Jing-Chen Peng / Kwin Shen

## Platform
Runs on Raspberry Pi 4

Uses ME 461 class platform (2-wheeled robot) with some modifications (mounted pi, mounted speaker, mounted camera, fixed wheel level)

## CS 445 side tasks (building 2d map using rgb camera + encoder data):

Lots of exploring around to see exactly what would work.
Settled on simple heuristic -- Using canny edge detector to grab edges, and looking for those closest to the bottom of the image.
Output from canny edge detector is fed through a Hough transform to (crudely) detect lines.
Lines are further combined using DBSCAN clustering algorithm in an attempt to reduce noise.
Lines are projected onto the ground plane as walls, with some heuristic "confidence levels"
based on how far away they are from the robot.
Global map is updated based on observation of walls (or lack thereof), based on
estimated visibility (assuming all walls are connected, and that the space between self and walls is visible.)
In each time step, the previous time step's detected lines are re-projected into the current time step's image
using dead reckoning data (and masked with the output of the canny edge detector), which helps reduce noise in the detected lines.

"wall-ness" of pixels is tracked by brightness of pixels (basically ends up being a discretized 2d map)
Turns out the same concepts that we used in class (masks mostly) can be used in mapping

Algorithm originally implemented in python with opencv, ran at like 2fps, too slow; moved to c++ and got a 10x speedup so that was cool.
Also learned C++ version of opencv in the meantime (along with basics of boost, eigen).

Note: Initially I thought I would try to use the lines themselves as part of my map,
to get a continuous-space map instead of a discretized one; haven't really looked into this further though...
Maybe I can do something like a second round of clustering using the lines projected onto the ground plane?

## ME 461 side tasks

Not shown: Code that runs on the redboard for low level control of the robot, and dead reckoning

Lots and lots of glue code.
    - HTTP server running on the pi to serve a GUI interface for users, and serve state info about the robot to other programs
    - Shared memory buffer for sharing images (probably not ideal to stream images to self with multiple processes)
