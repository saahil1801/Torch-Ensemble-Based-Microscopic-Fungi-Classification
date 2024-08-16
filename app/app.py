from flask import Flask, request, jsonify
from flasgger import Swagger
from models import models, preprocess_image, predict_ensemble
from dataset import get_transforms , FungiDataset
from config import config
app = Flask(__name__)
swagger = Swagger(app)

@app.route('/')
def home():
    return "Welcome to the Microscopic Fungi Prediction API", 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the class of a microscopic fungi image
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
    responses:
      200:
        description: The predicted class
        schema:
          type: object
          properties:
            predicted_class:
              type: string
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file:
        # Save the image to a temporary location
        temp_path = './temp_image.jpg'
        file.save(temp_path)
        
        # Preprocess the image
        image = preprocess_image(temp_path, get_transforms(config))
        dataset = FungiDataset(root_dir=config['base_dir'], transform=get_transforms(config))
        # Predict the class using the ensemble of models
        predicted_class_idx = predict_ensemble(models, image)
        predicted_class = dataset.classes[predicted_class_idx]
        
        return jsonify({"predicted_class": predicted_class})
    else:
        return jsonify({"error": "Error processing the image"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5007)
