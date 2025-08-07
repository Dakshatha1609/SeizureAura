# models/build_clean_model.py

import tensorflow as tf
from seizure_model import SeizurePredictionModel

input_shape = (500, 46)

model = SeizurePredictionModel(input_shape=input_shape)
model.build(input_shape=(None, *input_shape))

# Save without optimizer/dtype
model.save("seizure_model_clean.keras", include_optimizer=False)
print(" Clean model saved as seizure_model_clean.keras")
