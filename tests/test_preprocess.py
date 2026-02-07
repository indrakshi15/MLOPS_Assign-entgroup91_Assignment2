from PIL import Image
from src.preprocess import preprocess_image

def test_preprocess_image_shape():
    img = Image.new("RGB", (300, 300))
    output = preprocess_image(img)
    assert output.shape == (1, 224, 224, 3)

def test_preprocess_image_range():
    img = Image.new("RGB", (224, 224))
    output = preprocess_image(img)
    assert output.min() >= 0.0
    assert output.max() <= 1.0
