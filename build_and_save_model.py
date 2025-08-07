# build_and_save_model.py (place in root, same level as app.py)

import numpy as np
import tensorflow as tf

# Add models/ to sys.path so import works
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

from seizure_model import SeizurePredictionModel

# Define input shape (e.g., 100 timesteps × 46 features)
input_shape = (100, 46)

# Instantiate and build the model
model = SeizurePredictionModel(input_shape=input_shape)
dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
_ = model(dummy_input)  # Call once to initialize weights

# Save the model with weights (no optimizer)
model.save("seizure_model_cleaned.keras", include_optimizer=False)

print(" Fresh model saved as 'seizure_model_cleaned.keras'")
