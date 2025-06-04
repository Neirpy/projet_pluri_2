import cv2
import mediapipe as mp
import datetime # Import the datetime module
import os # Import the os module for path operations

"""
This script captures video from the webcam and uses MediaPipe to detect human pose landmarks.
It draws the detected landmarks and connections on the video feed.
Press 'q' or 'esc' to exit the video feed.
Press 'a' to save the landmarks to a file named 'landmarks_A_YYYYMMDD_HHMMSS.csv'.
Press 'b' to save the landmarks to a file named 'landmarks_B_YYYYMMDD_HHMMSS.csv'.
Parameters are given below, including the filename where to save the results (csv, with X,Y coordinates of landmarks).
"""

# parameters
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 1
video_source = 0  # Default webcam
# output_file is now dynamically generated
# end parameters


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_source)

landmarks_c = (234, 63, 247)
connection_c = 240  # (117,249,77),
thickness = 3
circle_r = 2
opened = True
tosave_droite = []
tosave_gauche = []
tosave_tourner_droite = []
tosave_tourner_gauche = []
tosave_avant = []
tosave_arriere = []
tosave_surprise = []
tosave_neutre = []
tosave_tete1 = []
tosave_tete2 = []
tosave_tete3 = []


def create_directory_if_not_exists(directory_path):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def keepLandmarks(landmarks, tosave_list):
    """
    Adds pose landmarks to the given list. If the list is empty, it adds a header row.
    """
    if not landmarks:
        print("No landmarks detected to save.")
        return

    if len(tosave_list) == 0:
        header = []
        for i in range(len(landmarks.landmark)):
            header.append(f"{mp_pose.PoseLandmark(i).name}_X")
            header.append(f"{mp_pose.PoseLandmark(i).name}_Y")
        tosave_list.append(header)
    line = []
    for l in landmarks.landmark:
        line.append(str(l.x))
        line.append(str(l.y))
    tosave_list.append(line)
    print("Added to list of landmarks to save.")

def saveLandmarks(tosave_list, file_prefix):
    """
    Saves the collected landmarks to a CSV file with a timestamp in a specific directory.
    """
    if not tosave_list:
        print(f"No landmarks to save for {file_prefix}.")
        return

    # Create the directory if it doesn't exist
    create_directory_if_not_exists(file_prefix)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("IA/data/",file_prefix, f"{file_prefix}_{timestamp}.csv") # Save inside the directory
    with open(filename, "w") as f:
        for line in tosave_list:
            f.write(",".join(line) + "\n")
    print(f"Saved landmarks to {filename}")

with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                  min_tracking_confidence=min_tracking_confidence,
                  model_complexity=model_complexity) as pose:
    while opened:
        opened, image = cap.read()
        if not opened:
            print("Failed to grab frame.")
            break

        # Flip the image horizontally for a natural selfie-view display.
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb) # Process the RGB image

        output_img = image.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(output_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(landmarks_c, thickness, circle_r),
                                     mp_drawing.DrawingSpec(connection_c, thickness, circle_r))
        cv2.imshow("Media pipe pose detection", output_img)
        key = cv2.waitKey(10)

        if key & 0xFF == ord("d"):
            keepLandmarks(results.pose_landmarks, tosave_droite)
        elif key & 0xFF == ord("q"):
            keepLandmarks(results.pose_landmarks, tosave_gauche)
        elif key & 0xFF == ord("m"):
            keepLandmarks(results.pose_landmarks, tosave_tourner_droite)
        elif key & 0xFF == ord("l"):
            keepLandmarks(results.pose_landmarks, tosave_tourner_gauche)
        elif key & 0xFF == ord("z"):
            keepLandmarks(results.pose_landmarks, tosave_avant)
        elif key & 0xFF == ord("s"):
            keepLandmarks(results.pose_landmarks, tosave_arriere)
        elif key & 0xFF == ord("a"):
            keepLandmarks(results.pose_landmarks, tosave_surprise)
        elif key & 0xFF == ord("n"):
            keepLandmarks(results.pose_landmarks, tosave_neutre)
        elif key & 0xFF == ord("1"):
            keepLandmarks(results.pose_landmarks, tosave_tete1)
        elif key & 0xFF == ord("2"):
            keepLandmarks(results.pose_landmarks, tosave_tete2)
        elif key & 0xFF == ord("3"):
            keepLandmarks(results.pose_landmarks, tosave_tete3)
        elif key & 0xFF == ord("p") or key == 27:  # 'q' or 'esc'
            break

cap.release()
cv2.destroyAllWindows()

# Save the collected landmarks for each key press at the end
saveLandmarks(tosave_gauche, "gauche")
saveLandmarks(tosave_droite, "droite")
saveLandmarks(tosave_tourner_gauche, "tourner_gauche")
saveLandmarks(tosave_tourner_droite, "tourner_droite")
saveLandmarks(tosave_avant, "avant")
saveLandmarks(tosave_arriere, "arriere")
saveLandmarks(tosave_surprise, "surprise")
saveLandmarks(tosave_neutre, "neutre")
saveLandmarks(tosave_tete1, "lent")
saveLandmarks(tosave_tete2, "moyen")
saveLandmarks(tosave_tete3, "rapide")