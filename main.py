from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import tensorflow as tf
import logging

# Setup logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load model and set image size
try:
    secondary_model = tf.keras.models.load_model("model.h5")
    IMG_SIZE = (224, 224)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model.")
    raise e

# Class mapping (you can customize friendly labels)
class_names = [
    "Apple_scab", "Apple_black_rot", "Apple_cedar_apple_rust", "Apple_healthy",
    "Blueberry_healthy", "Cherry_powdery_mildew", "Cherry_healthy",
    "Corn_gray_leaf_spot", "Corn_common_rust", "Corn_northern_leaf_blight", "Corn_healthy",
    "Grape_black_rot", "Grape_black_measles", "Grape_leaf_blight", "Grape_healthy",
    "Orange_haunglongbing", "Peach_bacterial_spot", "Peach_healthy",
    "Pepper_bacterial_spot", "Pepper_healthy",
    "Potato_early_blight", "Potato_healthy", "Potato_late_blight",
    "Raspberry_healthy", "Soybean_healthy", "Squash_powdery_mildew",
    "Strawberry_healthy", "Strawberry_leaf_scorch",
    "Tomato_bacterial_spot", "Tomato_early_blight", "Tomato_healthy",
    "Tomato_late_blight", "Tomato_leaf_mold", "Tomato_septoria_leaf_spot",
    "Tomato_spider_mites_two-spotted_spider_mite", "Tomato_target_spot",
    "Tomato_mosaic_virus", "Tomato_yellow_leaf_curl_virus"
]

# Optional friendly labels — modify as needed
class_mapping = {name: name.replace("_", " ") for name in class_names}

# Initialize app
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/check_image")
async def check_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.debug("Image loaded and converted to RGB.")

        image = image.resize(IMG_SIZE)
        image_array = np.array(image) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)
        logger.debug(f"Image shape after preprocessing: {image_batch.shape}")

        prediction = secondary_model.predict(image_batch)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = class_names[predicted_index]
        friendly_label = class_mapping.get(predicted_label, "Unknown")
        confidence = float(prediction[predicted_index])

        logger.info(f"Prediction: {predicted_label} -> {friendly_label} (Confidence: {confidence:.2f})")

        return JSONResponse({
            "result": friendly_label,
            "confidence": f"{confidence:.2f}",
            "raw_prediction": prediction.tolist(),
            "predicted_index": int(predicted_index),
            "predicted_class": predicted_label
        })

    except UnidentifiedImageError:
        logger.warning("Uploaded file is not a valid image.")
        raise HTTPException(status_code=400, detail="Invalid image format.")
    except Exception as e:
        logger.exception("Error during image classification.")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))
