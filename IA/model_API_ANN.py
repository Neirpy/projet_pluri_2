from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


from exploration_data.preprocessing import normalize_predict

# Load models and label encoders
model_move = tf.keras.models.load_model("models_ANN/best_ann_model_move.h5")
model_speed = joblib.load("models_temp/best_model_speed.pkl")

le_move = joblib.load("models_ANN/label_encoder_move.pkl")
le_speed = joblib.load("models_temp/label_encoder_speed.pkl")

# Index of the 'NEUTRE' class in the move label encoder
neutral_class_index_move = list(le_move.classes_).index("NEUTRE")

print("Index de la classe 'neutre' (move):", neutral_class_index_move)

# Define the features used for speed and move predictions
list_features_speed = [
    "NOSE_X", "NOSE_Y",
    "LEFT_EYE_INNER_X", "LEFT_EYE_INNER_Y",
    "LEFT_EYE_X", "LEFT_EYE_Y",
    "LEFT_EYE_OUTER_X", "LEFT_EYE_OUTER_Y",
    "RIGHT_EYE_INNER_X", "RIGHT_EYE_INNER_Y",
    "RIGHT_EYE_X", "RIGHT_EYE_Y",
    "RIGHT_EYE_OUTER_X", "RIGHT_EYE_OUTER_Y",
    "LEFT_EAR_X", "LEFT_EAR_Y",
    "RIGHT_EAR_X", "RIGHT_EAR_Y",
    "MOUTH_LEFT_X", "MOUTH_LEFT_Y",
    "MOUTH_RIGHT_X", "MOUTH_RIGHT_Y",
    "LEFT_SHOULDER_X", "LEFT_SHOULDER_Y",
    "RIGHT_SHOULDER_X", "RIGHT_SHOULDER_Y",
]

list_features_move = [
    "NOSE_X", "NOSE_Y",
    "LEFT_EAR_X", "LEFT_EAR_Y",
    "RIGHT_EAR_X", "RIGHT_EAR_Y",

    "LEFT_SHOULDER_X", "LEFT_SHOULDER_Y",
    "RIGHT_SHOULDER_X", "RIGHT_SHOULDER_Y",
    "LEFT_HIP_X", "LEFT_HIP_Y",
    "RIGHT_HIP_X", "RIGHT_HIP_Y",

    "LEFT_ELBOW_X", "LEFT_ELBOW_Y",
    "RIGHT_ELBOW_X", "RIGHT_ELBOW_Y",
    "LEFT_WRIST_X", "LEFT_WRIST_Y",
    "RIGHT_WRIST_X", "RIGHT_WRIST_Y",
    "LEFT_PINKY_X", "LEFT_PINKY_Y",
    "RIGHT_PINKY_X", "RIGHT_PINKY_Y",
    "LEFT_INDEX_X", "LEFT_INDEX_Y",
    "RIGHT_INDEX_X", "RIGHT_INDEX_Y",
    "LEFT_THUMB_X", "LEFT_THUMB_Y",
    "RIGHT_THUMB_X", "RIGHT_THUMB_Y",
]

# Ensure 'columns' matches the order and number of landmarks MediaPipe provides
columns = [
    "NOSE_X", "NOSE_Y", "LEFT_EYE_INNER_X", "LEFT_EYE_INNER_Y", "LEFT_EYE_X", "LEFT_EYE_Y",
    "LEFT_EYE_OUTER_X", "LEFT_EYE_OUTER_Y", "RIGHT_EYE_INNER_X", "RIGHT_EYE_INNER_Y", "RIGHT_EYE_X",
    "RIGHT_EYE_Y", "RIGHT_EYE_OUTER_X", "RIGHT_EYE_OUTER_Y", "LEFT_EAR_X", "LEFT_EAR_Y", "RIGHT_EAR_X",
    "RIGHT_EAR_Y", "MOUTH_LEFT_X", "MOUTH_LEFT_Y", "MOUTH_RIGHT_X", "MOUTH_RIGHT_Y", "LEFT_SHOULDER_X",
    "LEFT_SHOULDER_Y", "RIGHT_SHOULDER_X", "RIGHT_SHOULDER_Y", "LEFT_ELBOW_X", "LEFT_ELBOW_Y",
    "RIGHT_ELBOW_X", "RIGHT_ELBOW_Y", "LEFT_WRIST_X", "LEFT_WRIST_Y", "RIGHT_WRIST_X", "RIGHT_WRIST_Y",
    "LEFT_PINKY_X", "LEFT_PINKY_Y", "RIGHT_PINKY_X", "RIGHT_PINKY_Y", "LEFT_INDEX_X", "LEFT_INDEX_Y",
    "RIGHT_INDEX_X", "RIGHT_INDEX_Y", "LEFT_THUMB_X", "LEFT_THUMB_Y", "RIGHT_THUMB_X", "RIGHT_THUMB_Y",
    "LEFT_HIP_X", "LEFT_HIP_Y", "RIGHT_HIP_X", "RIGHT_HIP_Y", "LEFT_KNEE_X", "LEFT_KNEE_Y",
    "RIGHT_KNEE_X", "RIGHT_KNEE_Y", "LEFT_ANKLE_X", "LEFT_ANKLE_Y", "RIGHT_ANKLE_X", "RIGHT_ANKLE_Y",
    "LEFT_HEEL_X", "LEFT_HEEL_Y", "RIGHT_HEEL_X", "RIGHT_HEEL_Y", "LEFT_FOOT_INDEX_X",
    "LEFT_FOOT_INDEX_Y", "RIGHT_FOOT_INDEX_X", "RIGHT_FOOT_INDEX_Y"
]

# Entry class for input data
class InputData(BaseModel):
    values: List[float]
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
def predict(input: InputData):
    """
    Predict speed and move based on input values (mediapipe landmarks).
    """
    if len(input.values) != len(columns):
        return ["MOYEN","NEUTRE"]

    df_full = pd.DataFrame([input.values], columns=columns)
    df_full_norm = normalize_predict(df_full)
    df_full_norm = pd.DataFrame(df_full_norm, columns=columns)

    df_speed_norm = df_full_norm[list_features_speed]
    df_move_norm = df_full_norm[list_features_move]

    pred_speed_encoded = model_speed.predict(df_speed_norm)[0]

    df_move_norm = df_move_norm.astype('float32')

    # Prédictions for move : modèle Keras -> probability
    probas_move = model_move.predict(df_move_norm)

    # Apply custom logic to get the final prediction
    delta = 0.20
    alpha = 0.30

    pred_move_encoded = apply_custom_logic(probas_move, delta, alpha, neutral_class_index_move)[0]

    pred_speed_label = le_speed.inverse_transform([pred_speed_encoded])[0]
    pred_move_label = le_move.inverse_transform([pred_move_encoded])[0]

    print(f"Predicted speed: {pred_speed_label}, Predicted move: {pred_move_label}")

    return [pred_speed_label, pred_move_label]

# function to apply custom logic for move predictions
def apply_custom_logic(probas: np.ndarray, delta: float, alpha: float, neutral_class_index: int):
    """
    probas: (N_samples, N_classes) sorties softmax
    delta, alpha: seuils
    neutral_class_index: index de la classe 'neutre' (ex: 4)
    """
    p_neutral_all = probas[:, neutral_class_index]

    sorted_indices = np.argsort(probas, axis=1)
    p_sorted = np.take_along_axis(probas, sorted_indices, axis=1)
    p_max_all = p_sorted[:, -1]
    p_2nd_all = p_sorted[:, -2]

    delta_all = p_max_all - p_2nd_all

    preds_argmax = np.argmax(probas, axis=1)
    to_neutral = np.logical_or(p_neutral_all >= alpha, delta_all < delta)
    final_preds = preds_argmax.copy()
    final_preds[to_neutral] = neutral_class_index
    return final_preds
