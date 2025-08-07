# fix_model_clean_save.py

from tensorflow.keras.models import load_model, save_model
from keras.utils import custom_object_scope
from models.seizure_model import SeizurePredictionModel

# Load with scope
with custom_object_scope({'SeizurePredictionModel': SeizurePredictionModel}):
    model = load_model("seizure_model_fixed.keras", compile=False)

# Re-compile (empty), remove optimizer traces
model.compile()
save_model(model, "seizure_model_final.keras", include_optimizer=False)

print(" Final clean model saved: seizure_model_final.keras")
