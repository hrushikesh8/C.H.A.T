import os
import cv2
import numpy as np
import gdown
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.security import OAuth2PasswordBearer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import normalize
import tensorflow as tf
from tensorflow.keras import backend as K
from io import BytesIO

# Hardcoded token (for development/testing purposes only)
# AUTH_TOKEN = "test123token"

# Google Drive model file IDs
SEGMENTATION_MODEL_URL = "https://drive.google.com/file/d/1lR0iGnt0dVLd2MmVpXH3CRQSv030d7lW/view?usp=sharing"
CLASSIFICATION_MODEL_URL = "https://drive.google.com/file/d/1PUoZUAJt3w0R9ncC86A2sGA6p3T1y5cj/view?usp=sharing"

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

# Function to load image from uploaded file
# Function to load image from uploaded file
def loadImageFromFile(file: UploadFile):
    try:
        contents = file.file.read()
        img_array = np.asarray(bytearray(contents), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed, image is None!")
        img = cv2.resize(img, (256, 256))
        print("Loaded image shape:", img.shape, "Mean pixel value:", np.mean(img))
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Image Loading Error: {e}")
        return {"Error": "Invalid Image File"}

# Function to extract segmented region
def getCroppedImage(y, img):
    try:
        y = (y[0] * 255).astype("uint8")
        _, th1 = cv2.threshold(y, 12, 255, cv2.THRESH_BINARY)
        mask = th1 / 255.0
        cropped = np.where(mask[..., None] == 1, img[0], 255)
        print("Segmentation mask mean:", np.mean(mask))
        print("Cropped image mean:", np.mean(cropped))
        return cropped.astype(np.uint8)
    except Exception as e:
        print(f"Segmentation Error: {e}")
        return {"Error": "Segmentation Error"}

# Classification function
def anemiaResult(cropped_img):
    try:
        cropped_img = cv2.resize(cropped_img, (224, 224))
        cropped_img = np.expand_dims(cropped_img, axis=0)
        cropped_img = normalize(cropped_img, axis=-1)  # Normalize across channels
        print("Input to classifier shape:", cropped_img.shape)
        predictions = classification_model.predict(cropped_img)
        print("Prediction probabilities:", predictions)
        predicted_class = np.argmax(predictions)
        print("Predicted Class Index:", predicted_class)
        return {"Anemia_Status": CLASS_LABELS[predicted_class]}
    except Exception as e:
        print(f"Classification Error: {e}")
        return {"Error": "Classification Error"}


# Processing function
def parseImageFile(file: UploadFile):
    try:
        img = loadImageFromFile(file)
        if isinstance(img, dict):
            return img
        y = segmentation_model.predict(img)
        cropped_img = getCroppedImage(y, img)
        if isinstance(cropped_img, dict):
            return cropped_img
        return anemiaResult(cropped_img)
    except Exception as e:
        print(f"Processing Error: {e}")
        return {"Error": "Processing Error"}

# FastAPI Endpoints
@app.get("/")
def index():
    return "Anemia Detection API is running!"

# @app.post("/predict")
# def predict_anemia(file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
#     if token != AUTH_TOKEN:
#         return {"Error": "Invalid User"}
#     return parseImageFile(file)

@app.post("/predict")
def predict_anemia(file: UploadFile = File(...)):
    return parseImageFile(file)

@app.get("/reload_models")
def reload_models():
    global segmentation_model, classification_model
    segmentation_model = download_model(SEGMENTATION_MODEL_URL, "segmentation_model.h5")
    classification_model = download_model(CLASSIFICATION_MODEL_URL, "classification_model.h5")
    return {"message": "Models reloaded successfully!"}
