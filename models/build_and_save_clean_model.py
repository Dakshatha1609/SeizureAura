# build_and_save_clean_model.py

import tensorflow as tf
#from models.seizure_model import SeizurePredictionModel
from seizure_model import SeizurePredictionModel

# Define expected input shape
input_shape = (500, 46)

# Initialize and build the model
model = SeizurePredictionModel(input_shape=input_shape)

# Call the model ONCE to build its layers and variables
dummy_input = tf.random.normal((1, *input_shape))
model(dummy_input)

# Save using tf.saved_model (lowest-level, safest format)
model.save("seizure_model_cleaned.keras", include_optimizer=False, save_format="keras")

print("CLEAN model saved as 'seizure_model_cleaned.keras'")
