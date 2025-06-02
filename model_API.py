# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Charger le modèle une seule fois au démarrage
model = joblib.load("model.pkl")
app = FastAPI()

# 2. Définir le format d'entrée pour la prédiction
class PredictRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float


# 3. Endpoint de prédiction
@app.post("/predict")
def predict(input: PredictRequest):
    # Transformer les données en tableau NumPy pour sklearn
    features = np.array([[input.feature1, input.feature2, input.feature3, input.feature4]])

    # Faire la prédiction
    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}
