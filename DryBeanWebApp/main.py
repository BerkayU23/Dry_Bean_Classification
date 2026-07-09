import torch
from torch import nn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

# 1. Pydantic Model for Input Validation
class BeanFeatures(BaseModel):
    Area: float
    Perimeter: float
    MajorAxisLength: float
    MinorAxisLength: float
    AspectRation: float
    Eccentricity: float
    ConvexArea: float
    EquivDiameter: float
    Extent: float
    Solidity: float
    roundness: float
    Compactness: float
    ShapeFactor1: float
    ShapeFactor2: float
    ShapeFactor3: float
    ShapeFactor4: float


class MultiClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack_layer = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )

    def forward(self, x):
        return self.linear_stack_layer(x)


app = FastAPI(title="Dry Bean Classifier API")


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


model = None
scaler = None
label_encoder = None

@app.on_event("startup")
def load_assets():
    global model, scaler, label_encoder
    
    # Load Scaler
    try:
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        print("Warning: scaler.pkl not found. Please run export_assets.py to generate it.")
        
    # Load Label Encoder
    try:
        label_encoder = joblib.load("label_encoder.pkl")
    except FileNotFoundError:
        print("Warning: label_encoder.pkl not found. Please run export_assets.py to generate it.")
        
    # Load Model
    model_paths = ["DryBeanMC/DryBeanClassifier.pth", "DryBeanClassifier.pth"]
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            model = MultiClassifier()
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            model.eval()
            model_found = True
            print(f"Model loaded successfully from {path}")
            break
            
    if not model_found:
        print("Warning: DryBeanClassifier.pth not found. Model predictions will not work.")

@app.get("/")
def read_index():
    
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "index.html not found in static folder."}

@app.post("/predict")
def predict(features: BeanFeatures):
    if model is None or scaler is None or label_encoder is None:
        raise HTTPException(
            status_code=500, 
            detail="Model, Scaler veya LabelEncoder yüklenemedi. Lütfen gerekli dosyaları klasöre ekleyin."
        )
        

    feature_values = [
        features.Area, features.Perimeter, features.MajorAxisLength,
        features.MinorAxisLength, features.AspectRation, features.Eccentricity,
        features.ConvexArea, features.EquivDiameter, features.Extent,
        features.Solidity, features.roundness, features.Compactness,
        features.ShapeFactor1, features.ShapeFactor2, features.ShapeFactor3,
        features.ShapeFactor4
    ]
    
    X_input = np.array([feature_values])
    
    # Scale features
    X_scaled = scaler.transform(X_input)
    
    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Predict
    with torch.inference_mode():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        pred_idx = probs.argmax().item()
        
    # Inverse transform to get class name
    predicted_class = label_encoder.inverse_transform([pred_idx])[0]
    
    # Generate probabilities dictionary
    probabilities = {str(cls): float(prob) for cls, prob in zip(label_encoder.classes_, probs)}
    
    return {
        "class": predicted_class,
        "probabilities": probabilities
    }
