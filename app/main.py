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
    secondary_model = tf.keras.models.load_model("my_keras_model.keras")
    IMG_SIZE = (224, 224)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model.")
    raise e

# Correct class names based on index order (0 to 37)
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Friendly label mapping
class_mapping = {name: name.replace("_", " ") for name in class_names}

# Class label counts
class_label_counts = {
    "Apple___Apple_scab": 630,
    "Apple___Black_rot": 621,
    "Apple___Cedar_apple_rust": 275,
    "Apple___healthy": 1645,
    "Blueberry___healthy": 1502,
    "Cherry_(including_sour)___Powdery_mildew": 1052,
    "Cherry_(including_sour)___healthy": 854,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 513,
    "Corn_(maize)___Common_rust_": 1192,
    "Corn_(maize)___Northern_Leaf_Blight": 985,
    "Corn_(maize)___healthy": 1162,
    "Grape___Black_rot": 1180,
    "Grape___Esca_(Black_Measles)": 1383,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 1076,
    "Grape___healthy": 423,
    "Orange___Haunglongbing_(Citrus_greening)": 5507,
    "Peach___Bacterial_spot": 2297,
    "Peach___healthy": 360,
    "Pepper,_bell___Bacterial_spot": 997,
    "Pepper,_bell___healthy": 1478,
    "Potato___Early_blight": 1000,
    "Potato___Late_blight": 1000,
    "Potato___healthy": 152,
    "Raspberry___healthy": 371,
    "Soybean___healthy": 5090,
    "Squash___Powdery_mildew": 1835,
    "Strawberry___Leaf_scorch": 1109,
    "Strawberry___healthy": 456,
    "Tomato___Bacterial_spot": 2127,
    "Tomato___Early_blight": 1000,
    "Tomato___Late_blight": 1909,
    "Tomato___Leaf_Mold": 952,
    "Tomato___Septoria_leaf_spot": 1771,
    "Tomato___Spider_mites Two-spotted_spider_mite": 1676,
    "Tomato___Target_Spot": 1404,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 5357,
    "Tomato___Tomato_mosaic_virus": 373,
    "Tomato___healthy": 1591
}

# FastAPI app setup
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

        logger.debug("All class scores:")
        for idx, score in enumerate(prediction):
            logger.debug(f"{idx}: {class_names[idx]} -> {score:.4f}")

        predicted_index = int(np.argmax(prediction))
        predicted_label = class_names[predicted_index]
        friendly_label = class_mapping.get(predicted_label, "Unknown")
        confidence = float(prediction[predicted_index])

        logger.info(f"Prediction: {predicted_label} -> {friendly_label} (Confidence: {confidence:.2f})")

        return JSONResponse({
            "result": friendly_label,
            "confidence": f"{confidence:.2f}",
            "raw_prediction": prediction.tolist(),
            "predicted_index": predicted_index,
            "predicted_class": predicted_label,
            "class_label_counts": class_label_counts
        })

    except UnidentifiedImageError:
        logger.warning("Uploaded file is not a valid image.")
        raise HTTPException(status_code=400, detail="Invalid image format.")
    except Exception as e:
        logger.exception("Error during image classification.")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))
