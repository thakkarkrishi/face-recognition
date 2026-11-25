
import os
import numpy as np
import joblib
from PIL import Image

# -------------------- SETTINGS --------------------

MODEL_FILE = r"D:\data science\face recognition\face_recognition_pca_model.pkl"

IMAGE_FOLDER = r"D:\data science\face recognition"

# -------------------- LOAD MODEL --------------------

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Trained model not found. Run train_model.py first.")

saved_data = joblib.load(MODEL_FILE)

model = saved_data["model"]

image_size = saved_data["image_size"]

print("Model loaded successfully!")

# -------------------- SELECT IMAGE --------------------

images = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(images) == 0:
    raise FileNotFoundError(
        "No face images found in: " + IMAGE_FOLDER
    )

print("\nAvailable Images:")

for i, img in enumerate(images):
    print(i + 1, ".", img)

# Select image

choice = int(input("\nEnter image number for prediction: "))

if choice < 1 or choice > len(images):
    raise ValueError("Invalid image number!")

IMAGE_PATH = os.path.join(IMAGE_FOLDER, images[choice - 1])

print("\nSelected Image:", IMAGE_PATH)

# -------------------- LOAD IMAGE --------------------

image = Image.open(IMAGE_PATH).convert("L")

image = image.resize(tuple(image_size))

image = np.asarray(image, dtype=np.float32) / 255.0

new_face = image.flatten().reshape(1, -1)

# -------------------- PREDICTION --------------------

prediction = model.predict(new_face)[0]

probabilities = model.predict_proba(new_face)[0]

confidence = np.max(probabilities) * 100

# -------------------- DISPLAY RESULT --------------------

print("\n========== FACE RECOGNITION RESULT ==========")

print("Predicted Person:", prediction)

print("Prediction Confidence:", round(confidence, 2), "%")

print("\n==============================================")

# -------------------- DISPLAY IMAGE --------------------

import matplotlib.pyplot as plt

plt.imshow(image, cmap="gray")

plt.title("Predicted Person: " + str(prediction))

plt.axis("off")

plt.show()
