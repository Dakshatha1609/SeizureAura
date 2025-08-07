from seizure_model import SeizurePredictionModel
from tensorflow.keras.models import save_model
import numpy as np

# Create dummy input with shape (1, 500, 46)
dummy_input = np.random.rand(1, 500, 46)

# Build model
model = SeizurePredictionModel(input_shape=(500, 46))
model(dummy_input)  # Trigger model building

# Save without optimizer
save_model(model, "seizure_model_clean.keras", include_optimizer=False)

print(" Clean model saved as seizure_model_clean.keras")
