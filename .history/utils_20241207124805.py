import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
import json

def process_image(image_path):
    """Preprocess the input image for prediction."""
    img = load_img(image_path, target_size=(224, 224))  
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def load_label_map(category_names_path):
    """Load the label map from a JSON file."""
    with open(category_names_path, "r") as f:
        return json.load(f)
