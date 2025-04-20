import os
import cv2
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from tensorflow.keras.models import load_model

IMG_SIZE = 224

# Paths to your models
MODEL_PATHS = {
    "hand": "hand_fracture_model.h5",
    "cervical": "cervical_fracture_model.h5"
}

# Upload folder setup
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models into dictionary
models = {
    "hand": load_model(MODEL_PATHS["hand"]),
    "cervical": load_model(MODEL_PATHS["cervical"])
}

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_image(image_path, model_name):
    model = models[model_name]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    label = "Fractured" if np.argmax(prediction) == 1 else "Normal"
    confidence = float(np.max(prediction)) * 100
    return label, confidence

def retrain_model(image_path, correct_label, model_name):
    model = models[model_name]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array([[1, 0]] if correct_label == "Normal" else [[0, 1]])
    model.fit(img, y, epochs=1, verbose=0)
    model.save(MODEL_PATHS[model_name])

html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Medical Predictor</title>
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
            background: #f3f4f6;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        .sidebar {
            width: 200px;
            background-color: #004d40;
            color: white;
            padding: 50px 20px 20px 20px;
            transition: width 0.3s ease;
        }

        .sidebar h3 {
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.2rem;
        }

        .sidebar button {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #00695c;
            border: none;
            color: white;
            border-radius: 5px;
            cursor: pointer;
        }

        .sidebar button:hover {
            background-color: #00796b;
        }

        .toggle-sidebar {
            position: fixed;
            top: 10px;
            left: 10px;
            background-color: #004d40;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 5px;
            cursor: pointer;
            z-index: 999;
        }

        .sidebar.collapsed {
            width: 0;
            padding: 0;
            overflow: hidden;
        }

        .container {
            flex-grow: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .card {
            width: 850px;
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            text-align: center;
        }

        h2 {
            color: #004d40;
        }

        #drop-area {
            border: 2px dashed #004d40;
            padding: 30px;
            border-radius: 10px;
            cursor: pointer;
            background-color: #f0f0f0;
        }

        #drop-area:hover {
            background-color: #e0f2f1;
        }

        input[type="file"] {
            display: none;
        }

        .horizontal {
            display: flex;
            justify-content: space-around;
            margin-top: 30px;
            gap: 20px;
        }

        #preview {
            max-width: 250px;
            max-height: 250px;
            border-radius: 10px;
            display: none;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }

        .btn {
            padding: 10px 15px;
            background-color: #00796b;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 20px;
        }

        .btn:hover {
            background-color: #004d40;
        }

        #loader {
            display: none;
            margin: 20px auto;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #00796b;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }

        #resultBox {
            display: flex;
	    justify-content: center;
	    text-align: center;
            padding: 20px;
            border: 2px solid #004d40;
            border-radius: 10px;
            background-color: #e0f2f1;
            color: #004d40;
            font-weight: bold;
            font-size: 24px;
            min-width: 200px;
        }

        #newUploadBtn {
            display: none;
        }

        .robot {
            margin-top: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .robot-message {
            margin-top: 10px;
            font-size: 16px;
            font-weight: 500;
            color: #004d40;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="sidebar" id="sidebar">
        <h3>Select Model</h3>
        <button onclick="setModel('hand')">🖐 Hand</button>
        <button onclick="setModel('cervical')">🩻 Cervical</button>
    </div>

    <button class="toggle-sidebar" onclick="toggleSidebar()">☰</button>

    <div class="container">
        <div class="card">
            <h2 id="header">😇😇😇</h2>
            <div id="uploadSection" style="display: none;">
                <div id="drop-area" onclick="fileInput.click();">
                    Click or Drop X-ray Here
                    <input type="file" id="fileInput" accept="image/*">
                </div>
            </div>

            <div id="loader"></div>

            <div class="horizontal">
                <img id="preview" src="">
                <div id="resultBox">Select Model to Start</div>
            </div>

            <div class="robot" id="robotContainer">
                <lottie-player
                    src="https://assets2.lottiefiles.com/packages/lf20_3vbOcw.json"
                    background="transparent"
                    speed="1"
                    style="width: 500px; height: 500px;"
                    loop
                    autoplay>
                </lottie-player>
                <div class="robot-message" id="robotMessage">🤖 Welcome To The X-Ray Prediction !</div>
            </div>

            <button class="btn" id="newUploadBtn" onclick="resetPage()">🔄 New Upload</button>
        </div>
    </div>

    <script>
        let currentModel = null;
        const fileInput = document.getElementById("fileInput");
        const preview = document.getElementById("preview");
        const resultBox = document.getElementById("resultBox");
        const loader = document.getElementById("loader");
        const header = document.getElementById("header");
        const uploadSection = document.getElementById("uploadSection");
        const newUploadBtn = document.getElementById("newUploadBtn");
        const robotContainer = document.getElementById("robotContainer");
        const robotMessage = document.getElementById("robotMessage");

        function setModel(model) {
            currentModel = model;
            header.innerText = model === 'hand' ? "🖐 Hand X-ray Upload" : "🩻 Cervical X-ray Upload";
            uploadSection.style.display = "block";
            resetPage();
        }

        fileInput.addEventListener("change", () => {
            const file = fileInput.files[0];
            if (!file || !currentModel) return;

            const reader = new FileReader();
            reader.onload = () => {
                preview.src = reader.result;
                preview.style.display = "block";
            };
            reader.readAsDataURL(file);

            const formData = new FormData();
            formData.append("file", file);
            formData.append("model", currentModel);

            loader.style.display = "inline-block";
            resultBox.style.display = "none";
            robotContainer.style.display = "none";

            fetch("/predict", {
                method: "POST",
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                loader.style.display = "none";
                resultBox.style.display = "block";
                resultBox.innerHTML = `Prediction: ${data.label}<br>Accuracy: ${data.confidence}`;           
                newUploadBtn.style.display = "inline-block";

                // Show animated robot message
                robotContainer.style.display = "flex";
                robotContainer.innerHTML = `
                    <img src="https://i.gifer.com/7efs.gif" width="80" />
                    <div class="robot-message">🤖 I think it's ${data.label}</div>
                `;
                document.getElementById("drop-area").style.display = "none";
            })
            .catch(() => {
                loader.style.display = "none";
                resultBox.style.display = "block";
                resultBox.textContent = "❌ Prediction failed.";
                robotMessage.textContent = "🤖 Oops! Something went wrong.";
            });
        });

        function resetPage() {
            fileInput.value = "";
            preview.src = "";
            preview.style.display = "none";
            resultBox.style.display = "none";
            newUploadBtn.style.display = "none";
            robotContainer.style.display = "flex";
            robotContainer.innerHTML = `
                <lottie-player
                    src="https://assets2.lottiefiles.com/packages/lf20_3vbOcw.json"
                    background="transparent"
                    speed="1"
                    style="width: 300px; height: 300px;"
                    loop
                    autoplay>
                </lottie-player>
                <div class="robot-message" id="robotMessage">🤖 Please select the X-Ray to be uploaded.</div>
            `;
            document.getElementById("drop-area").style.display = "block";
        }

        function toggleSidebar() {
            const sidebar = document.getElementById("sidebar");
            sidebar.classList.toggle("collapsed");
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'Missing file or model selection'})

    file = request.files['file']
    model_name = request.form['model']

    if file.filename == '' or model_name not in models:
        return jsonify({'error': 'Invalid file or model'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    label, confidence = predict_image(filepath, model_name)
    return jsonify({'label': label, 'confidence': f"{confidence:.2f}%"})

@app.route('/feedback', methods=['POST'])
def feedback():
    if 'file' not in request.files or 'label' not in request.form or 'model' not in request.form:
        return jsonify({'error': 'Missing file, label or model'})

    file = request.files['file']
    label = request.form['label']
    model_name = request.form['model']

    if model_name not in models:
        return jsonify({'error': 'Invalid model'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    retrain_model(filepath, label, model_name)
    return jsonify({'message': '✅ Feedback received and model updated!'})

if __name__ == '__main__':
    app.run(debug=True)
