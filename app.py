from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np, json, uvicorn, os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model and remedies
print("Loading the trained model...")

try:
    # Try to load the trained model with actual weights first
    model = load_model("model/tomato_leaf_disease_model_trained.keras", compile=False)
    print("SUCCESS: Loaded trained model with actual weights!")
    
except Exception as e1:
    print(f"Could not load trained model: {e1}")
    
    try:
        # Try to load the trained SavedModel format
        model = load_model("model/tomato_leaf_disease_model_trained_savedmodel", compile=False)
        print("SUCCESS: Loaded trained SavedModel format!")
        
    except Exception as e2:
        print(f"Could not load trained SavedModel: {e2}")
        
        try:
            # Fallback: Try the compatible model without trained weights
            model = load_model("model/tomato_leaf_disease_model.keras", compile=False)
            print("WARNING: Loaded model without trained weights - predictions may be inaccurate")
            
        except Exception as e3:
            print(f"Could not load compatible model: {e3}")
            
            try:
                # Last resort: Try original H5 with custom objects
                class LegacyInputLayer:
                    def __init__(self, **kwargs):
                        kwargs.pop('batch_shape', None)
                        self.input_shape = kwargs.get('input_shape', (128, 128, 3))
                        self.name = kwargs.get('name', 'input_layer')
                
                custom_objects = {'InputLayer': LegacyInputLayer}
                model = load_model("model/tomato_leaf_disease_model.h5", 
                                 compile=False, 
                                 custom_objects=custom_objects)
                print("SUCCESS: Loaded original H5 model with custom objects!")
                
            except Exception as e4:
                print(f"All model loading attempts failed: {e4}")
                print("Creating a fallback model...")
                
                # Create a simple CNN as last resort
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
                
                model = Sequential([
                    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                    MaxPooling2D(2, 2),
                    Conv2D(64, (3, 3), activation='relu'),
                    MaxPooling2D(2, 2),
                    Conv2D(128, (3, 3), activation='relu'),
                    MaxPooling2D(2, 2),
                    Flatten(),
                    Dense(128, activation='relu'),  # Match the trained model architecture
                    Dropout(0.5),
                    Dense(10, activation='softmax')
                ])
                
                print("WARNING: Using fallback model - predictions will be random")
                print("INFO: The model architecture is ready but needs training")
with open("remedies.json") as f:
    remedies = json.load(f)

class_labels = list(remedies.keys())

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Save uploaded file
    img_path = f"static/{file.filename}"
    with open(img_path, "wb") as f:
        f.write(await file.read())

    # Prepare image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    disease = class_labels[idx]
    confidence = float(np.max(pred) * 100)
    advice = remedies.get(disease, "No advice available.")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": {
            "disease": disease,
            "confidence": f"{confidence:.2f}%",
            "advice": advice,
            "img_path": img_path
        }
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4000)
