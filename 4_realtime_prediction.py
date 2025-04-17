import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model_path = "model/stress_detection_model.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: Model file not found at {model_path}. Please train the model first.")

model = load_model(model_path)
print("Model loaded successfully.")

# Simulated test image (Replace this with actual face image input)
face = np.random.rand(1, 48, 48, 1)  # Incorrect shape (1, 48, 48, 1)

# ✅ Resize the image to match the model's expected input shape (64, 64, 1)
face_resized = cv2.resize(face[0], (64, 64))  # Remove batch dimension, resize
face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension back
face_resized = np.expand_dims(face_resized, axis=-1)  # Ensure shape is (1, 64, 64, 1)

# Make prediction
prediction = model.predict(face_resized)
predicted_class = np.argmax(prediction)

# Output result
print(f"Predicted class: {predicted_class}, Confidence: {prediction[0][predicted_class]:.4f}")
