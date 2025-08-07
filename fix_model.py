# fix_model.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope
from models.seizure_model import SeizurePredictionModel

# Fix DTypePolicy issue by including it explicitly in the scope
with custom_object_scope({
    "SeizurePredictionModel": SeizurePredictionModel,
    "DTypePolicy": tf.keras.mixed_precision.Policy  #  Important fix
}):
    model = load_model("seizure_model_fixed.keras", compile=False)

# Save cleaned model without dtype policy
model.save("seizure_model_cleaned.keras", include_optimizer=False)

print(" Cleaned model saved as 'seizure_model_cleaned.keras'")
