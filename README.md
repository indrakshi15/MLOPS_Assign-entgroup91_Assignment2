CATS vs DOGS – END-TO-END MLOPS PIPELINE

Project Overview
This project implements a complete end-to-end MLOps pipeline for a binary image classification model (Cats vs Dogs).
It demonstrates model training, versioning, containerization, CI/CD automation, deployment, monitoring, and post-deployment evaluation using open-source tools.

Use Case
Input: 224x224 RGB image
Output: Cat or Dog
Dataset: Kaggle Cats vs Dogs Dataset
Data Split: 80% Train / 10% Validation / 10% Test
Data Augmentation applied for better generalization.

M1 – Model Development & Experiment Tracking

CNN model built using TensorFlow/Keras

Experiments tracked using MLflow (accuracy, loss, artifacts)

Dataset and trained model versioned using DVC

M2 – Containerized Inference Service
Inference API built using FastAPI.

Available Endpoints:
GET /health
Returns service status.

POST /predict
Accepts image file and returns prediction, probability, and latency.

GET /metrics
Returns total request count and average inference latency.

The model is packaged inside a Docker container for reproducible deployment.

M3 – Continuous Integration (CI)
Implemented using GitHub Actions.
On every push to the main branch:

Install dependencies

Run unit tests (pytest)

Pull Docker image

Deploy container

Execute smoke test

M4 – Continuous Deployment (CD)
On updates to the main branch:

Pull latest Docker image from Docker Hub

Replace running container

Validate deployment using health endpoint

M5 – Monitoring & Post-Deployment Evaluation

Basic Monitoring & Logging:

Structured logging of prediction requests

Request counter

Latency tracking

/metrics endpoint to expose runtime statistics

Post-Deployment Model Evaluation:
A labeled batch of images is sent to the deployed API using an evaluation script to compute post-deployment accuracy.
This verifies deployment correctness and model performance consistency.

Project Structure
api/
src/
models/
evaluation/
tests/
Dockerfile
requirements.txt
.github/workflows/ci.yml

Technology Stack
TensorFlow / Keras
MLflow
DVC
FastAPI
Docker
GitHub Actions
Docker Hub

How to Run Locally

Start container:
docker run -p 8000:8000 <docker-username>/cats-dogs-api:latest

Health check:
http://localhost:8000/health

Make prediction using curl:
curl -X POST http://localhost:8000/predict
 -F "file=@dog.jpg"

Outcome
This project demonstrates a production-style ML lifecycle including reproducibility, automation, deployment, monitoring, and post-deployment validation.
