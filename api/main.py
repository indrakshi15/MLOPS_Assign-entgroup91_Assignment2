from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import io

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_cnn_cats_dogs.h5")

print("MODEL PATH:", MODEL_PATH)

# Load trained model
model = load_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "UP"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        image = image.resize((224, 224))
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict
        preds = model.predict(image)

        # Handle different output shapes safely
        if preds.ndim == 2 and preds.shape[1] == 1:
            prob = float(preds[0][0])
        elif preds.ndim == 2 and preds.shape[1] == 2:
            prob = float(preds[0][1])  # dog probability
        else:
            prob = float(preds.flatten()[0])

        label = "Cat" if prob > 0.5 else "Dog"


        return {
            "prediction": label,
            "probability": prob
        }

    except Exception as e:
        return {
            "error": str(e),
            "type": str(type(e))
        }
