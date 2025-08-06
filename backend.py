from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from keras.saving import register_keras_serializable
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "seizure_model.keras"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO)

@register_keras_serializable()
class SeizurePredictionModel(tf.keras.Model):
    def __init__(self, input_shape=None, **kwargs):
        super(SeizurePredictionModel, self).__init__(**kwargs)
        self.input_shape_ = input_shape

        self.cnn = models.Sequential([
            layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3)
        ])
        self.lstm = layers.Bidirectional(layers.LSTM(64))
        self.dense = layers.Dense(64, activation='relu')
        self.out = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.cnn(x)
        x = self.lstm(x)
        x = self.dense(x)
        return self.out(x)

    def get_config(self):
        config = super().get_config()
        config.update({"input_shape": self.input_shape_})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

try:
    model = load_model(MODEL_PATH, custom_objects={"SeizurePredictionModel": SeizurePredictionModel})
    logging.info(" Seizure prediction model loaded successfully.")
except Exception as e:
    logging.error(f" Failed to load model: {e}")
    model = None

def preprocess_eeg(file_path):
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=[np.number]).dropna()

    if df.empty:
        raise ValueError("Uploaded file contains no valid numeric data after cleaning.")

    data = df.values
    if data.shape[1] > data.shape[0]:
        data = data.T

    if data.shape[1] > 46:
        data = data[:, :46]

    if data.shape[1] != 46:
        raise ValueError(f"Model expects 46 features, found {data.shape[1]}")

    data = data.reshape(1, data.shape[0], data.shape[1])
    return data

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logging.info(f" Received file: {file.filename}")

        data = preprocess_eeg(file_path)

        if not model:
            return {"error": "Model not loaded. Please check logs."}

        prediction = model.predict(data)[0][0]
        result = "Seizure Risk" if prediction > 0.5 else "No Seizure Risk"
        logging.info(f" Prediction: {result} (prob={prediction:.4f})")

        return {"result": result, "probability": float(prediction)}

    except Exception as e:
        logging.error(f" Prediction error: {e}")
        return {"error": str(e)}
