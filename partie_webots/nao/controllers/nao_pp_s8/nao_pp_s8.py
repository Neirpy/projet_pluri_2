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

pose_vec=['MOYEN', 'NEUTRE']
current_speed = 2
while robot.step(timestep) != -1:
    # a processed version of vec should go as input of the model
    print(pose_vec)
    speed = pose_vec[0]
    if speed == 'LENT' and current_speed != 1:
        current_speed -=1
        motion.change_speed(current_speed)
        
    if speed == 'RAPIDE' and current_speed != 3:
        current_speed += 1
        motion.change_speed(current_speed)
    print(current_speed)
    # this code just show the different moves
    if pose_vec[1] == 'NEUTRE':
        motion.neutral(current_speed)
    if pose_vec[1] == 'TOURNER_DROITE':
        motion.turn("left", current_speed)
    if pose_vec[1] == 'TOURNER_GAUCHE':
        motion.turn("right", current_speed)
    if pose_vec[1] == 'AVANT': 
        motion.forward(current_speed)
    if pose_vec[1] == 'ARRIERE':
        motion.backward(current_speed)
    if pose_vec[1] == 'DROITE':
        if current_speed == 3 :
            current_speed = 2
        motion.sidestep("left", current_speed)
    if pose_vec[1] == 'GAUCHE': 
        if current_speed == 3 :
            current_speed = 2
        motion.sidestep("right", current_speed)
    if pose_vec[1] == 'SURPRISE':
        motion.coucou(current_speed)
    #speed = (speed % 3) + 1 
    if motion.can_get_anim:
        #pose_vec=["coucou", 2]
        pose_vec = pose.getPose()
    # here is where the model can be plugged
   