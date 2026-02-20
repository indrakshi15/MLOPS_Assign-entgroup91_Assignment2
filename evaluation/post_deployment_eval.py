import requests
import os

API_URL = "http://localhost:8000/predict"

test_folder = "sample_test"

correct = 0
total = 0

for label in ["Cat", "Dog"]:
    folder_path = os.path.join(test_folder, label)

    for img in os.listdir(folder_path):
        with open(os.path.join(folder_path, img), "rb") as f:
            response = requests.post(API_URL, files={"file": f})
            pred = response.json()["prediction"]

            if pred == label:
                correct += 1
            total += 1

accuracy = correct / total
print("Post-deployment accuracy:", accuracy)