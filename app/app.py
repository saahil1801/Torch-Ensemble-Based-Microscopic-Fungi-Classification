from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dataset import get_transforms, FungiDataset
from config import config
from funcs.models import models, preprocess_image, predict_ensemble
import uvicorn

app = FastAPI(
    title="Microscopic Fungi Prediction API",
    description="This API allows users to upload images of microscopic fungi and receive predictions on the class of fungi using an ensemble of trained models.",
    version="1.0.0"
)

class PredictionResponse(BaseModel):
    predicted_class: str

@app.get("/", summary="Home", tags=["Home"])
def home():
    """
    Welcome message for the API.
    """
    return {"message": "Welcome to the Microscopic Fungi Prediction API"}

@app.post("/predict", response_model=PredictionResponse, summary="Predict Fungi Class", tags=["Prediction"])
async def predict(image: UploadFile = File(...)):
    """
    Predict the class of a microscopic fungi image.
    
    - **image**: Upload a microscopic fungi image file for prediction.
    
    Returns the predicted class of the fungi.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image file provided")

    try:
        # Save the image to a temporary location
        temp_path = './temp_image.jpg'
        with open(temp_path, "wb") as buffer:
            buffer.write(await image.read())

        # Preprocess the image
        transforms = get_transforms(config)
        preprocessed_image = preprocess_image(temp_path, transforms)
        dataset = FungiDataset(root_dir=config['base_dir'], transform=transforms)

        # Predict the class using the ensemble of models
        predicted_class_idx = predict_ensemble(models, preprocessed_image)
        predicted_class = dataset.classes[predicted_class_idx]

        return {"predicted_class": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")

# Command to run the app: uvicorn app:app --host 0.0.0.0 --port 8080 / http://localhost:8080/docs

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
