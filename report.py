import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Constants
IMG_SIZE = 224
DATA_DIR = "hand_fracture"
MODEL_PATH = "hand_fracture_model.h5"

# Load and preprocess the data (same as before)
def load_data():
    X, y = [], []
    for label in ["fracture", "normal"]:
        path = os.path.join(DATA_DIR, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(1 if label == "fracture" else 0)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Generate classification report
def generate_report():
    X_train, X_val, y_train, y_val = load_data()
    model = load_model(MODEL_PATH)
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["Normal", "Fracture"]))

generate_report()
