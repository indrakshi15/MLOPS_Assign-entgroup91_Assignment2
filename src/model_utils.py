def get_label_from_prob(prob: float):
    return "Dog" if prob >= 0.5 else "Cat"
