PROJECT OVERVIEW

This project implements a complete end-to-end MLOps pipeline for a binary image classification model (Cats vs Dogs).

The system includes:

->Model training

->Experiment tracking (MLflow)

->Data & model versioning (DVC)

->FastAPI inference service

->Docker containerization

->CI/CD automation (GitHub Actions)

->Monitoring & post-deployment evaluation

**PREREQUISITES**

Install the following tools:

Python 3.10

Git

Docker Desktop

DVC

MLflow 

**Verify installation:**

 python --version
 
docker --version 

git --version 

dvc --version 

mlflow --version

**STEP 1 – CLONE REPOSITORY**

git clone <repository-url>
cd cats_dogs_mlops


STEP 2 – CREATE VIRTUAL ENVIRONMENT

Windows:

python -m venv venv
venv\Scripts\activate

Mac/Linux:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

STEP 3 – DATASET SETUP

Download the Kaggle Cats vs Dogs dataset.

Place dataset inside:

data/cats_and_dogs/

If using DVC:

dvc pull

This restores the versioned dataset and model.

STEP 4 – TRAIN THE MODEL (M1)

Run:

python src/train.py

This will:

Train CNN model

Save model inside models/

**Log experiments to MLflow**

To view MLflow UI:

From your project root directory run

mlflow ui

Open browser:

http://localhost:5000

STEP 5 – RUN INFERENCE API LOCALLY (M2)

Start FastAPI service:

uvicorn api.main:app --host 0.0.0.0 --port 8000

**Test health endpoint:**

http://localhost:8000/health
**
Test prediction:**

curl -X POST http://localhost:8000/predict
 -F "file=@dog.jpg"

STEP 6 – BUILD DOCKER IMAGE

docker build -t indrakshi/cats-dogs-api:latest .

Push to Docker Hub:

docker login
docker push <docker-username>/cats-dogs-api:latest

STEP 7 – RUN CONTAINER

docker run -p 8000:8000 <docker-username>/cats-dogs-api:latest

Verify:

http://localhost:8000/health

STEP 8 – CI/CD PIPELINE (M3 & M4)

CI/CD is configured using GitHub Actions.

On every push to main branch:

Dependencies installed

Unit tests executed

Docker image pulled

Container deployed

Smoke test executed

To trigger CI:

git add .
git commit -m "Trigger pipeline"
git push

Check GitHub → Actions tab.

STEP 9 – MONITORING (M5)

Make prediction requests.

Then check metrics:

http://localhost:8000/metrics

Metrics include:

Total requests

Average latency

Logs display prediction label, probability, and latency.

STEP 10 – POST-DEPLOYMENT EVALUATION

Create folder:

sample_test/
├── Cat/
├── Dog/

Place a few labeled images.

Run:

python evaluation/post_deployment_eval.py

This computes post-deployment accuracy.

PROJECT STRUCTURE

api/ – FastAPI inference service
src/ – Model training code
models/ – Trained model (DVC tracked)
evaluation/ – Post-deployment evaluation script
tests/ – Unit tests
Dockerfile – Container definition
requirements.txt – Dependencies
.github/workflows/ci.yml – CI/CD pipeline



