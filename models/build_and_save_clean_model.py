# build_and_save_clean_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class SeizurePredictionModel(tf.keras.Model):
    def __init__(self, input_shape=(500, 46), **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = input_shape
        self.cnn = models.Sequential([
            layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
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

# Initialize model
model = SeizurePredictionModel()

# Build by calling once with dummy input (to initialize weights)
dummy_input = np.random.rand(1, 500, 46).astype(np.float32)
model(dummy_input)

# Save the model with initialized weights
model.save("seizure_model_cleaned.keras", include_optimizer=False)
print("Cleaned model with weights saved as 'seizure_model_cleaned.keras'")
