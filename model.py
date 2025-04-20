import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define constants
IMG_SIZE = 224  
DATA_DIR = "cervical fracture\train"
MODEL_PATH = "cervical_fracture_model.h5"

# Step 1: Load and preprocess images
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
    y = to_categorical(y, num_classes=2)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax")  
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Step 3: Train and save the model
def train_model():
    X_train, X_val, y_train, y_val = load_data()
    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
    model.save(MODEL_PATH)
    print("Model saved successfully as 'cervical_fracture_model.h5'")

# Step 4: Predict function for new images
def predict_fracture(image_path):
    model = load_model(MODEL_PATH)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    return "Fractured" if np.argmax(prediction) == 1 else "Normal"

# Train the model
train_model()

# Test the model on an image
test_image_path = "cervical fracture\test\sample.jpg"  # Change this path
print(f"Prediction: {predict_fracture(test_image_path)}")
