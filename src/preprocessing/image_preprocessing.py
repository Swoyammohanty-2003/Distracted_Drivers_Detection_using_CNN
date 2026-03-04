import cv2
import numpy as np
from src.config.config import IMG_SIZE

def preprocess_frame(frame):
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype('float32') / 255.0
    img_array = np.expand_dims(normalized, axis=0)
    return img_array
