IDD - F2017 Final Project: Attention Tracking for Drivers
==========================================================

A video demonstrating our project can be found here: https://www.youtube.com/watch?v=Y04UstcZIsw&t=55s

We used OpenCV and Dlib packages to track a users face for driving attention tracking. We tweaked this code to be used on a RaspberryPi Generation 3 B, please reach out if you are interested. 

Code Summary: The code runs in a simple loop where it searchs for and analyzes facial landmarks in every frame to run a state machine and provide feedback.

Required Packages:
------------------
- numpy
- argparse
- time
- datetime
- scipy
- cv2
- dlib
- imutils
- serial
- sys
- playsound

Our Subpackages:
----------------
- Clientdefs: for web client communication
- imdefs: image processing functions
- LEDDriver: Arduino Serial Communication

We would like to thank Adrian Rosebrock of PyImageSearch, who developed very similar algorithms in the same months of our project development. Also a huge thanks to Professor Bjoern Hartmann and GSI Rundong Tian for a smoothly run class.
