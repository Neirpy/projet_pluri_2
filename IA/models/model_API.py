# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")
app = FastAPI()

class PredictRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict")
def predict(input: PredictRequest):
    features = np.array([[input.feature1, input.feature2, input.feature3, input.feature4]])

    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}

@app.get("/random")
def random():
    random_choice_action = np.random.choice(["avant", "arriere", "droite", "gauche", "tourner_droite", "tourner_gauche", "coucou", "neutre"])
    random_choice_speed = np.random.choice(
        ["1", "2", "3"])
    return random_choice_action, random_choice_speed
