# clean_and_save_model_final.py

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils import custom_object_scope

# Define the model inline to avoid external references
class SeizurePredictionModel(tf.keras.Model):
    def __init__(self, input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = input_shape
        self.cnn = models.Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu'),
            layers.MaxPooling1D(2),
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

# Build and save model using custom_object_scope
with custom_object_scope({'SeizurePredictionModel': SeizurePredictionModel}):
    model = SeizurePredictionModel(input_shape=(500, 46))
    model.build(input_shape=(None, 500, 46))
    model.save("seizure_model_cleaned.keras", include_optimizer=False)

print(" Model rebuilt and saved as 'seizure_model_cleaned.keras'")
