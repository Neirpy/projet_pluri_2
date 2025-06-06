from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

from exploration_data.preprocessing import normalize_predict

# Chargement des modèles et des encodeurs
model_speed = joblib.load("models_temp/best_model_speed.pkl")
model_move = joblib.load("models_one_arm/best_model_one_arm_move.pkl")
le_move = joblib.load("models_one_arm/label_encoder_one_arm_move.pkl")
le_speed = joblib.load("models_temp/label_encoder_speed.pkl")

# Liste des features utilisées pour la vitesse et le mouvement
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

    "RIGHT_ELBOW_X", "RIGHT_ELBOW_Y",
    "RIGHT_WRIST_X", "RIGHT_WRIST_Y",
    "RIGHT_PINKY_X", "RIGHT_PINKY_Y",
    "RIGHT_INDEX_X", "RIGHT_INDEX_Y",
    "RIGHT_THUMB_X", "RIGHT_THUMB_Y",
]

# Liste complète des colonnes utilisées dans le DataFrame
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

# Classe d'entrée
class InputData(BaseModel):
    values: List[float]
app = FastAPI()

# Route pour la prédiction
@app.post("/predict")
def predict(input: InputData):
    """
    Prédit la vitesse et le mouvement à partir des données d'entrée.
    """
    if len(input.values) != len(columns):
        return ["MOYEN","NEUTRE"]

    # Création du DataFrame complet
    df_full = pd.DataFrame([input.values], columns=columns)

    # Normalisation de toutes les colonnes
    df_full_norm = normalize_predict(df_full)
    df_full_norm = pd.DataFrame(df_full_norm, columns=columns)

    # Sélection des features après normalisation
    df_speed_norm = df_full_norm[list_features_speed]
    df_move_norm = df_full_norm[list_features_move]

    # Prédictions
    pred_speed_encoded = model_speed.predict(df_speed_norm)[0]
    pred_move_encoded = model_move.predict(df_move_norm)[0]

    pred_speed_label = le_speed.inverse_transform([pred_speed_encoded])[0]
    pred_move_label = le_move.inverse_transform([pred_move_encoded])[0]

    print(f"Predicted speed: {pred_speed_label}, Predicted move: {pred_move_label}")
    return [
        pred_speed_label,
        pred_move_label
        ]

# Route pour une action aléatoire
@app.get("/random")
def random():
    random_choice_action = np.random.choice(["AVANT", "ARRIERE", "DROITE", "GAUCHE", "TOURNER_DROITE", "TOURNER_GAUCHE", "COUCOU", "NEUTRE"])
    random_choice_speed = np.random.choice(
        ["LENT", "MOYEN", "RAPIDE"])
    return random_choice_speed,random_choice_action
