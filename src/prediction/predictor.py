import tensorflow as tf
import numpy as np
from src.config.config import CLASSES_DICT

class Predictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, img_array):
        prediction = self.model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(prediction))
        label = CLASSES_DICT[predicted_class]
        return predicted_class, label
