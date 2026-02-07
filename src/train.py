
"""
Standalone training script for Cats vs Dogs
(Exported from Jupyter notebook for MLOps compliance)
"""

import tensorflow as tf
import mlflow
import mlflow.tensorflow

IMG_SIZE = 224

mlflow.set_experiment("Cats_vs_Dogs_M1")

with mlflow.start_run():

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Model saved for deployment
    model.save("../models/baseline_cnn_cats_dogs.h5")
