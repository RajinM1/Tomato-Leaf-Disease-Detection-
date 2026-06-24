# Tomato Leaf Disease Detection

A FastAPI web application that detects common tomato leaf diseases from an uploaded leaf image and returns a predicted disease, confidence score, and treatment advice.

## Overview

Tomato Leaf Disease Detection helps users identify tomato plant leaf conditions using a trained deep learning model. The app provides a simple browser interface where users can upload a tomato leaf image and receive an instant prediction with a recommended remedy.

## Features

- Upload tomato leaf images through a web interface
- Predict 10 tomato leaf classes, including healthy leaves
- Display prediction confidence
- Show disease-specific treatment recommendations
- Uses TensorFlow/Keras model files
- Built with FastAPI, Jinja2 templates, HTML, CSS, and JavaScript

## Disease Classes

The app supports the following tomato leaf categories:

- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites / Two-spotted Spider Mite
- Tomato Target Spot
- Tomato Mosaic Virus
- Tomato Yellow Leaf Curl Virus
- Tomato Healthy

## Project Structure

```text
tomato_leaf_doctor/
|-- app.py
|-- requirements.txt
|-- remedies.json
|-- model/
|   |-- tomato_leaf_disease_model.h5
|   |-- tomato_leaf_disease_model.keras
|   |-- tomato_leaf_disease_model_trained.keras
|   |-- tomato_leaf_disease_model_savedmodel/
|   `-- tomato_leaf_disease_model_trained_savedmodel/
|-- static/
|   |-- style.css
|   |-- script.js
|   |-- logo.png
|   `-- placeholder_leaf.jpg
|-- templates/
|   `-- index.html
|-- convert_model.py
|-- fix_model.py
`-- load_trained_model.py
```

## Tech Stack

- Python
- FastAPI
- TensorFlow / Keras
- NumPy
- Pillow
- Jinja2
- HTML, CSS, JavaScript

## Setup Instructions

### 1. Clone The Repository

```bash
git clone https://github.com/RajinM1/Tomato-Leaf-Disease-Detection-.git
cd Tomato-Leaf-Disease-Detection-
```

### 2. Create A Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run The App

```bash
python app.py
```

Open the app in your browser:

```text
http://127.0.0.1:4000
```

You can also run it with Uvicorn:

```bash
uvicorn app:app --host 127.0.0.1 --port 4000 --reload
```

## How It Works

1. The user uploads a tomato leaf image.
2. The app saves the uploaded image temporarily inside the `static/` folder.
3. The image is resized to `128x128` pixels.
4. Pixel values are normalized before prediction.
5. The Keras model predicts the most likely disease class.
6. The app displays the disease name, confidence score, uploaded image, and remedy.

## Model Loading Order

The app tries to load model files in this order:

1. `model/tomato_leaf_disease_model_trained.keras`
2. `model/tomato_leaf_disease_model_trained_savedmodel/`
3. `model/tomato_leaf_disease_model.keras`
4. `model/tomato_leaf_disease_model.h5`
5. A fallback CNN architecture if all model files fail to load

For best results, make sure the trained model files are present in the `model/` directory.

## Example Usage

1. Start the server.
2. Open `http://127.0.0.1:4000`.
3. Choose a tomato leaf image.
4. Click **Analyze Leaf**.
5. View the predicted disease, confidence, and treatment recommendation.

## Notes

- This project is intended for educational and research use.
- Predictions depend on the quality of the uploaded image and the trained model.
- For real farming decisions, confirm severe plant disease issues with an agricultural expert.
- Uploaded images are currently saved into the `static/` directory.

## Author

Built by Rajin.
