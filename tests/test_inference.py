from src.model_utils import get_label_from_prob

def test_dog_prediction():
    assert get_label_from_prob(0.9) == "Dog"

def test_cat_prediction():
    assert get_label_from_prob(0.1) == "Cat"
