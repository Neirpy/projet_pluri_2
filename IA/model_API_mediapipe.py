# main.py (Modified FastAPI Application)
from typing import List
import cv2
import mediapipe as mp
import joblib
import pandas as pd
from fastapi import FastAPI

from exploration_data.preprocessing import normalize_predict


# --- Load ML Models and Label Encoders ---
try:
    model_speed = joblib.load("models_temp/best_model_speed.pkl")
    model_move = joblib.load("models_temp/best_model_move.pkl")
    le_move = joblib.load("models_temp/label_encoder_move.pkl")
    le_speed = joblib.load("models_temp/label_encoder_speed.pkl")
    print("ML models and label encoders loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model files. Make sure they are in the 'models_temp' directory relative to main.py: {e}")
    # Consider exiting or raising an exception if models are critical
    exit(1)

# --- Define Feature Lists and All Columns (as per your original main.py) ---
list_features_speed = [
    "NOSE_X", "NOSE_Y", "LEFT_EYE_INNER_X", "LEFT_EYE_INNER_Y", "LEFT_EYE_X", "LEFT_EYE_Y",
    "LEFT_EYE_OUTER_X", "LEFT_EYE_OUTER_Y", "RIGHT_EYE_INNER_X", "RIGHT_EYE_INNER_Y", "RIGHT_EYE_X",
    "RIGHT_EYE_Y", "RIGHT_EYE_OUTER_X", "RIGHT_EYE_OUTER_Y", "LEFT_EAR_X", "LEFT_EAR_Y", "RIGHT_EAR_X",
    "RIGHT_EAR_Y", "MOUTH_LEFT_X", "MOUTH_LEFT_Y", "MOUTH_RIGHT_X", "MOUTH_RIGHT_Y", "LEFT_SHOULDER_X",
    "LEFT_SHOULDER_Y", "RIGHT_SHOULDER_X", "RIGHT_SHOULDER_Y",
]

list_features_move = [
    "NOSE_X", "NOSE_Y", "LEFT_EAR_X", "LEFT_EAR_Y", "RIGHT_EAR_X", "RIGHT_EAR_Y",
    "LEFT_SHOULDER_X", "LEFT_SHOULDER_Y", "RIGHT_SHOULDER_X", "RIGHT_SHOULDER_Y",
    "LEFT_HIP_X", "LEFT_HIP_Y", "RIGHT_HIP_X", "RIGHT_HIP_Y",
    "LEFT_ELBOW_X", "LEFT_ELBOW_Y", "RIGHT_ELBOW_X", "RIGHT_ELBOW_Y",
    "LEFT_WRIST_X", "LEFT_WRIST_Y", "RIGHT_WRIST_X", "RIGHT_WRIST_Y",
    "LEFT_PINKY_X", "LEFT_PINKY_Y", "RIGHT_PINKY_X", "RIGHT_PINKY_Y",
    "LEFT_INDEX_X", "LEFT_INDEX_Y", "RIGHT_INDEX_X", "RIGHT_INDEX_Y",
    "LEFT_THUMB_X", "LEFT_THUMB_Y", "RIGHT_THUMB_X", "RIGHT_THUMB_Y",
]

# Ensure 'columns' matches the order and number of landmarks MediaPipe provides (x, y for 33 landmarks = 66 values)
ALL_POSE_LANDMARKS = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE",
    "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
    "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY",
    "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP",
    "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]
columns = []
for landmark in ALL_POSE_LANDMARKS:
    columns.append(f"{landmark}_X")
    columns.append(f"{landmark}_Y")


# --- MediaPipe Pose Setup (now global for FastAPI) ---
# Initialize MediaPipe outside the request handler for efficiency
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1 # Adjust as needed
)

# Initialize OpenCV video capture (global so it stays open)
# Use 0 for default webcam, or provide a video file path
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam. Ensure it's not in use by another application.")
    # Consider raising an exception or handling gracefully
    # For now, exit if no camera, as it's critical for this setup.
    exit(1)

# FastAPI App Instance
app = FastAPI()

# Function to convert MediaPipe landmarks to your vector format
def _toVector(landmarks):
    if landmarks is None:
        return []
    line = []
    for l in landmarks.landmark:
        line.append(l.x)
        line.append(l.y)
    return line

@app.get("/get_prediction")
def get_prediction():
    """
    Captures an image, detects pose, predicts movement and speed,
    and returns the prediction.
    """
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        # If no frame, return a default or error state
        return {"speed": "MOYEN", "move": "NEUTRE", "message": "Failed to capture image"}

    # Flip the image horizontally for a mirror effect, and convert BGR to RGB.
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False # For performance
    results = pose_detector.process(image)
    image.flags.writeable = True

    vec = _toVector(results.pose_landmarks)

    if len(vec) == len(columns): # Check if we got all 66 landmark values
        try:
            # Create DataFrame for prediction
            df_full = pd.DataFrame([vec], columns=columns)

            # Normalize data
            df_full_norm = normalize_predict(df_full)
            df_full_norm = pd.DataFrame(df_full_norm, columns=columns)

            # Select features for speed and movement models
            df_speed_norm = df_full_norm[list_features_speed]
            df_move_norm = df_full_norm[list_features_move]

            # Make predictions
            pred_speed_encoded = model_speed.predict(df_speed_norm)[0]
            pred_move_encoded = model_move.predict(df_move_norm)[0]

            pred_speed_label = str(le_speed.inverse_transform([pred_speed_encoded])[0])
            pred_move_label = str(le_move.inverse_transform([pred_move_encoded])[0])

            print(f"Predicted speed: {pred_speed_label}, Predicted move: {pred_move_label}")
            return pred_speed_label, pred_move_label

        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"speed": "NEUTRE", "move": "NEUTRE", "message": f"Prediction error: {e}"}
    else:
        print("Not enough landmarks detected for prediction.")
        return {"speed": "NEUTRE", "move": "NEUTRE", "message": "No pose detected"}
