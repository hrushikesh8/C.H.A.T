import os
import cv2
import numpy as np
import gdown
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import normalize
import urllib.request
import tensorflow as tf
from tensorflow.keras import backend as K
# Hardcoded token (for development/testing purposes only)
AUTH_TOKEN = "test123token"



# Google Drive model file IDs
SEGMENTATION_MODEL_URL = os.getenv("SEGMENTATION_MODEL_URL")
CLASSIFICATION_MODEL_URL = os.getenv("CLASSFICATION_MODEL_URL")

# Labels for classification model
CLASS_LABELS = ['Mild Anemia', 'Moderate Anemia', 'No Anemia', 'Severe Anemia']

# Define Dice Loss function
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

# Download and load model with custom objects
def download_model(url, filename):
    gdown.download(url, filename, quiet=False)
    return load_model(filename, custom_objects={'dice_loss': dice_loss}, compile=False)

# Load models
segmentation_model = download_model(SEGMENTATION_MODEL_URL, "segmentation_model.h5")
classification_model = download_model(CLASSIFICATION_MODEL_URL, "classification_model.h5")

# FastAPI app
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Function to load image from URL
def loadImage(path):
    try:
        with urllib.request.urlopen(path) as response:
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
            img = cv2.resize(img, (256, 256))
            img = np.expand_dims(img, axis=0)
            return img
    except Exception as e:
        print(f"Image Loading Error: {e}")
        return {"Error": "Invalid Image URL"}


# Function to extract segmented region
def getCroppedImage(y, img):
    try:
        y = (y[0] * 255).astype("uint8")
        _, th1 = cv2.threshold(y, 12, 255, cv2.THRESH_BINARY)
        mask = th1 / 255.0  # keep float for masking
        cropped = np.where(mask[..., None] == 1, img[0], 255)
        return cropped.astype(np.uint8)
    except Exception as e:
        print(f"Segmentation Error: {e}")
        return {"Error": "Segmentation Error"}


# Classification function
def anemiaResult(cropped_img):
    try:
        cropped_img = cv2.resize(cropped_img, (224, 224))
        cropped_img = np.expand_dims(cropped_img, axis=0)
        cropped_img = normalize(cropped_img, axis=1)
        predictions = classification_model.predict(cropped_img)
        predicted_class = np.argmax(predictions)
        return {"Anemia_Status": CLASS_LABELS[predicted_class]}
    except Exception as e:
        print(f"Classification Error: {e}")
        return {"Error": "Classification Error"}


# Processing function
def parseImage(path):
    try:
        img = loadImage(path)
        if isinstance(img, dict):
            return img  # Return error if image loading failed
        y = segmentation_model.predict(img)
        cropped_img = getCroppedImage(y, img)
        if isinstance(cropped_img, dict):
            return cropped_img  # Return error if segmentation failed
        return anemiaResult(cropped_img)
    except Exception as e:
        print(f"Processing Error: {e}")
        return {"Error": "Processing Error"}

# FastAPI Endpoints
@app.get("/")
def index():
    return "Anemia Detection API is running!"

# @app.post("/predict")
# def predict_anemia(image_url: str, token: str = Depends(oauth2_scheme)):
#     if token != os.getenv("REQUEST_AUTHENTICATION"):  # Validate Token
#         return {"Error": "Invalid User"}
#     return parseImage(image_url)

@app.post("/predict")
def predict_anemia(image_url: str, token: str = Depends(oauth2_scheme)):
    if token != AUTH_TOKEN:  # Use the hardcoded token instead of environment variable
        return {"Error": "Invalid User"}
    return parseImage(image_url)


# @app.post("/predict")
# def predict_anemia(image_url: str):
#     return parseImage(image_url)


@app.get("/reload_models")
def reload_models():
    """Reload the models dynamically from Google Drive"""
    global segmentation_model, classification_model
    segmentation_model = download_model(SEGMENTATION_MODEL_URL, "segmentation_model.h5")
    classification_model = download_model(CLASSIFICATION_MODEL_URL, "classification_model.h5")
    return {"message": "Models reloaded successfully!"}
