# requires opencv 
#   pip insytall opencv-python
# and mediapipe
#  pip install mediapipe
from controller import Robot
from naomotion import NaoMotion
import cv2
from pose import Pose

robot = Robot()
timestep = int(robot.getBasicTimeStep())
motion = NaoMotion(robot)

# parameters
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 1
video_source = 0  # Default webcam
# end parameters

pose = Pose(min_detection_confidence, min_tracking_confidence, 
            model_complexity, video_source)

speed=3

# arriere = 1
# avant = 2
# droite = 3 
# gauche = 4
# tourner_droite = 5
# tourner_gauche = 6
# surprise = 7
# neutre = 8

pose_vec=["neutre", 1]
while robot.step(timestep) != -1:
    # a processed version of vec should go as input of the model
    print(pose_vec)
    speed = int(pose_vec[1])
    # this code just show the different moves
    if pose_vec[0] == 6:
        motion.turn("left", speed)
    if pose_vec[0] == 5:
        motion.turn("right", speed)
    if pose_vec[0] == 2: 
        motion.forward(speed)
    if pose_vec[0] == 1:
        motion.backward(speed)
    if pose_vec[0] == 4:
        if speed == 3 :
            speed = 2
        motion.sidestep("left", speed)
    if pose_vec[0] == 3: 
        if speed == 3 :
            speed = 2
        motion.sidestep("right", speed)
    if pose_vec[0] == 7:
        motion.coucou(speed)
    #speed = (speed % 3) + 1 
    if motion.can_get_anim:
        #pose_vec=["coucou", 2]
        pose_vec = pose.getPose()
    # here is where the model can be plugged
   