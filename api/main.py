from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import io
import time
import logging

app = FastAPI()

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cats-dogs-api")

# -----------------------
# Metrics Counters
# -----------------------
request_count = 0
total_latency = 0.0

# -----------------------
# Load Model
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_cnn_cats_dogs.h5")

logger.info(f"Loading model from {MODEL_PATH}")
model = load_model(MODEL_PATH)
logger.info("Model loaded successfully")

# -----------------------
# Health Endpoint
# -----------------------
@app.get("/health")
def health():
    return {"status": "UP"}

# -----------------------
# Metrics Endpoint
# -----------------------
@app.get("/metrics")
def metrics():
    global request_count, total_latency

    avg_latency = total_latency / request_count if request_count > 0 else 0

    return {
        "total_requests": request_count,
        "average_latency_ms": round(avg_latency * 1000, 2)
    }

# -----------------------
# Prediction Endpoint
# -----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count, total_latency

    start_time = time.time()

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

        if preds.ndim == 2 and preds.shape[1] == 1:
            prob = float(preds[0][0])
        elif preds.ndim == 2 and preds.shape[1] == 2:
            prob = float(preds[0][1])
        else:
            prob = float(preds.flatten()[0])

        label = "Cat" if prob > 0.5 else "Dog"

        latency = time.time() - start_time

        # Update metrics
        request_count += 1
        total_latency += latency

        logger.info(
            f"Prediction: {label} | Prob: {round(prob,3)} | "
            f"Latency: {round(latency*1000,2)} ms"
        )

        return {
            "prediction": label,
            "probability": prob,
            "latency_ms": round(latency * 1000, 2)
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {
            "error": str(e),
            "type": str(type(e))
        }