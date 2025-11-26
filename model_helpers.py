import os
import pickle
import numpy as np
from PIL import Image

def load_model_and_encoder(model_path="model.pkl", encoder_path="label_encoder.pkl"):
    model = None
    encoder = None
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded model from", model_path)
        except Exception as e:
            print("Failed to load model:", e)
            model = None
    if os.path.exists(encoder_path):
        try:
            with open(encoder_path, "rb") as f:
                encoder = pickle.load(f)
            print("Loaded label encoder from", encoder_path)
        except Exception as e:
            print("Failed to load label encoder:", e)
            encoder = None
    return model, encoder

def preprocess_image_for_model(pil_image: Image.Image, size=(224,224)):
    """
    Basic preprocess: resize, normalize, flatten.
    If your model expects a different preproc, replace this function.
    """
    img = pil_image.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    # Flatten for classical ML; for deep learners you may want (1,224,224,3)
    return arr.flatten()

def demo_prediction(image):
    """
    Return a demo prediction when no model is available.
    """
    labels = ["Tomato_Blight", "Powdery_Mildew"]
    choice = labels[0] if np.mean(image.resize((8,8)).convert("L")) < 120 else labels[1]
    confidence = round(float(np.random.uniform(0.70, 0.96)), 2)
    return {"label": choice, "confidence": confidence}
