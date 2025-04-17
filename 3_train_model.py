import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

# Simulated dataset (Replace with actual data)
X_data = np.random.rand(1000, 64, 64, 1)  # Example data (1000 images of 64x64 grayscale)
y_data = np.random.randint(0, 2, 1000)  # Example binary labels

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),  # FIXED: Adjusted to correct size
    Dense(512, activation='relu'),
    Dense(2, activation='softmax')  # Binary classification
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Ensure 'model/' directory exists
if not os.path.exists("model"):
    os.makedirs("model")

# Save trained model
model.save("model/stress_detection_model.h5")

print("Model training complete and saved successfully.")
