# models/build_and_save_clean_model.py

import tensorflow as tf
from seizure_model import SeizurePredictionModel

# Set the input shape
input_shape = (500, 46)  # Update as per your actual data shape

# Build the model
model = SeizurePredictionModel(input_shape=input_shape)
model.build(input_shape=(None, *input_shape))

# Save model without any optimizer or dtype info
model.save("seizure_model_cleaned.keras", include_optimizer=False)
print(" Cleaned model saved as 'seizure_model_cleaned.keras'")
