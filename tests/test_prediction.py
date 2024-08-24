import pytest
from funcs.models import models, preprocess_image, predict_ensemble
from dataset import FungiDataset
from config import config
from dataset import get_transforms, FungiDataset

@pytest.fixture
def sample_image_path():
    # Provide a path to a sample image file
    return 'archive-2/test/H6/H6_13a_4.jpg.jpg'

@pytest.fixture
def fungi_dataset():
    transforms = get_transforms(config)
    return FungiDataset(root_dir=config['base_dir'], transform=transforms)

@pytest.fixture
def preprocessed_image(sample_image_path):
    # Load transforms and preprocess the image
    transforms = get_transforms(config)
    return preprocess_image(sample_image_path, transforms)

def test_predicted_class_index(preprocessed_image, fungi_dataset):
    # Test that the predicted class index is within the valid range
    predicted_class_idx = predict_ensemble(models, preprocessed_image)
    
    assert 0 <= predicted_class_idx < len(fungi_dataset.classes), "Predicted class index is out of range"

def test_predicted_class_name(preprocessed_image, fungi_dataset):
    # Test that the predicted class name is correct
    predicted_class_idx = predict_ensemble(models, preprocessed_image)
    
    expected_class = "H6"  # Replace with the actual expected class name
    predicted_class = fungi_dataset.classes[predicted_class_idx]
    
    assert predicted_class == expected_class, f"Expected class '{expected_class}', got '{predicted_class}'"
