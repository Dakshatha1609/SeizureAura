# build_clean_model.py

import numpy as np
from tensorflow.keras.models import save_model
from models.seizure_model import SeizurePredictionModel

# Instantiate and build model
model = SeizurePredictionModel(input_shape=(500, 46))

# Trigger build with dummy input (to create weights)
dummy_input = np.random.rand(1, 500, 46).astype(np.float32)
model(dummy_input)

# Save clean model (no optimizer, no dtype policy)
save_model(model, "seizure_model_cleaned.keras", include_optimizer=False)

print(" Saved 'seizure_model_cleaned.keras' with no dtype or optimizer configs.")
