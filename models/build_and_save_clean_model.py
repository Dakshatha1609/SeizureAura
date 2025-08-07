# models/build_and_save_clean_model.py

from seizure_model import SeizurePredictionModel
import numpy as np
import tensorflow as tf

# Simulate dummy input data to build model
dummy_input = np.random.rand(1, 500, 46).astype(np.float32)

# Build model
model = SeizurePredictionModel(input_shape=(500, 46))
model(dummy_input)  # Force build

# Save clean model
model.save("seizure_model_cleaned.keras", include_optimizer=False)
print(" Cleaned model saved successfully as 'seizure_model_cleaned.keras'")
