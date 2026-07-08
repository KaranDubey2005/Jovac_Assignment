import io
import os
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np

app = FastAPI(title="Fashion Product Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

macro_mapping = {
    "T-shirt/top": "Tops & Outerwear",
    "Pullover": "Tops & Outerwear",
    "Dress": "Tops & Outerwear",
    "Coat": "Tops & Outerwear",
    "Shirt": "Tops & Outerwear",
    "Trouser": "Trousers & Bottoms",
    "Sandal": "Footwear",
    "Sneaker": "Footwear",
    "Ankle boot": "Footwear",
    "Bag": "Accessories & Bags"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"FastAPI using device: {device}")

def load_model():
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 10)
    
    model_path = "best_product_classifier.pkl"
    if not os.path.exists(model_path):
        model_path = "best_product_classifier.pth"
        
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model weight file not found. Please train the model first.")
        
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

try:
    classifier_model = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    classifier_model = None

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if classifier_model is None:
        raise HTTPException(status_code=500, detail="Classifier model is not loaded.")
        
    try:
        contents = await file.read()
        raw_image = Image.open(io.BytesIO(contents))
        
        gray_image = raw_image.convert('L')
        np_img = np.array(gray_image)
        
        h, w = np_img.shape
        border_pixels = np.concatenate([
            np_img[0, :],
            np_img[:, 0],
            np_img[:, -1]
        ])
        avg_border = np.median(border_pixels)
        
        np_img = np.abs(np_img.astype(np.float32) - avg_border).astype(np.uint8)
        
        gray_diff = Image.fromarray(np_img)
        max_dim = max(w, h)
        squared_image = Image.new('L', (max_dim, max_dim), 0)
        if w > h:
            squared_image.paste(gray_diff, (0, (max_dim - h) // 2))
        else:
            squared_image.paste(gray_diff, ((max_dim - w) // 2, 0))
            
        np_img = np.array(squared_image)
        
        margin = int(max_dim * 0.05)
        margin_bottom = int(max_dim * 0.05)
        np_img[:margin, :] = 0
        np_img[-margin_bottom:, :] = 0
        np_img[:, :margin] = 0
        np_img[:, -margin:] = 0
        
        np_img[np_img < 15] = 0
        
        min_val = np_img.min()
        max_val = np_img.max()
        if max_val > min_val:
            np_img = ((np_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            
        image = Image.fromarray(np_img).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
    try:
        image_mirrored = ImageOps.mirror(image)
        
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        input_tensor_mirrored = preprocess(image_mirrored).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = classifier_model(input_tensor)
            outputs_mirrored = classifier_model(input_tensor_mirrored)
            
            probs = torch.softmax(outputs, dim=1)[0]
            probs_mirrored = torch.softmax(outputs_mirrored, dim=1)[0]
            
            probabilities = (probs + probs_mirrored) / 2.0
            confidence, pred_class_idx = torch.max(probabilities, dim=0)
            
        scores = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        macro_scores = {}
        for fine_class, score in scores.items():
            macro_class = macro_mapping[fine_class]
            macro_scores[macro_class] = macro_scores.get(macro_class, 0.0) + score
            
        best_macro_class = max(macro_scores, key=macro_scores.get)
        best_macro_confidence = macro_scores[best_macro_class]
        
        sorted_macro_scores = sorted(macro_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_fine_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "predicted_macro": best_macro_class,
            "macro_confidence": best_macro_confidence,
            "predicted_fine": class_names[pred_class_idx.item()],
            "fine_confidence": float(confidence.item()),
            "macro_probabilities": dict(sorted_macro_scores),
            "fine_probabilities": dict(sorted_fine_scores)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/api/samples")
def get_samples():
    try:
        import pandas as pd
        import random
        import base64
        import numpy as np
        
        test_df = pd.read_csv("Data/fashion-mnist_test.csv")
        samples = []
        indices = random.sample(range(len(test_df)), 5)
        for idx in indices:
            row = test_df.iloc[idx]
            label = int(row.iloc[0])
            pixels = row.iloc[1:].values.astype(np.uint8).reshape(28, 28)
            
            pil_img = Image.fromarray(pixels).convert('L')
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            samples.append({
                "id": int(idx),
                "true_label": class_names[label],
                "image": f"data:image/png;base64,{img_str}"
            })
        return samples
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch samples: {str(e)}")

frontend_build_path = "frontend/dist"
if os.path.exists(frontend_build_path):
    app.mount("/", StaticFiles(directory=frontend_build_path, html=True), name="static")
else:
    print(f"Warning: '{frontend_build_path}' directory not found yet. Build the React app first.")
