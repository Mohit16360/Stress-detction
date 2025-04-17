import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define image size
IMG_SIZE = 64
data_path = "dataset"
categories = ["stressed", "neutral", "relaxed"]

X, y = [], []

# Load and preprocess images
for label, category in enumerate(categories):
    folder = os.path.join(data_path, category)

    if not os.path.exists(folder):
        print(f"Error: Folder {folder} does not exist.")
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        # Read and resize the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label)

# Convert lists to NumPy arrays
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# Normalize pixel values
X = X / 255.0

# Save data
np.save("X.npy", X)
np.save("y.npy", y)

print("Preprocessing complete. Data saved as X.npy and y.npy.")
