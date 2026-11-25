
import os
import random
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

# -------------------- SETTINGS --------------------

MODEL_FILE = "face_recognition_pca_model.pkl"

DATASET_FOLDER = "lfw_funneled"

NUMBER_OF_IMAGES = 10

# -------------------- LOAD MODEL --------------------

saved_data = joblib.load(MODEL_FILE)

model = saved_data["model"]

image_size = saved_data["image_size"]

class_names = saved_data["class_names"]

print("Model loaded successfully!")

# -------------------- LOAD IMAGES --------------------

image_paths = []
actual_labels = []

for person in class_names:

    person_folder = os.path.join(DATASET_FOLDER, person)

    if not os.path.isdir(person_folder):
        continue

    for image_file in os.listdir(person_folder):

        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):

            image_paths.append(
                os.path.join(person_folder, image_file)
            )

            actual_labels.append(person)

if len(image_paths) == 0:
    raise ValueError("No images found in the dataset!")

# -------------------- RANDOM IMAGE SELECTION --------------------

random.seed(42)

selected_indices = random.sample(
    range(len(image_paths)),
    min(NUMBER_OF_IMAGES, len(image_paths))
)

X_test = []
y_test = []
display_images = []

for index in selected_indices:

    image = Image.open(image_paths[index]).convert("L")

    image = image.resize(tuple(image_size))

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    X_test.append(image_array.flatten())

    y_test.append(actual_labels[index])

    display_images.append(image_array)

X_test = np.asarray(X_test)

# -------------------- PREDICTION --------------------

y_pred = model.predict(X_test)

# -------------------- ACCURACY --------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== FACE RECOGNITION RESULTS ==========")

print("Total Images Tested:", len(X_test))

print("Correct Predictions:", sum(
    actual == predicted
    for actual, predicted in zip(y_test, y_pred)
))

print("Incorrect Predictions:", sum(
    actual != predicted
    for actual, predicted in zip(y_test, y_pred)
))

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n========== PREDICTION DETAILS ==========")

for i in range(len(y_test)):

    print("\nImage:", i + 1)

    print("Actual Person:", y_test[i])

    print("Predicted Person:", y_pred[i])

    if y_test[i] == y_pred[i]:
        print("Result: Correct")
    else:
        print("Result: Incorrect")

# -------------------- CLASSIFICATION REPORT --------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# -------------------- DISPLAY RESULTS --------------------

plt.figure(figsize=(15, 8))

for i in range(len(X_test)):

    plt.subplot(2, 5, i + 1)

    plt.imshow(display_images[i], cmap="gray")

    if y_test[i] == y_pred[i]:
        color = "green"
    else:
        color = "red"

    plt.title(
        "Actual: " + y_test[i] +
        "\nPredicted: " + y_pred[i],
        color=color,
        fontsize=8
    )

    plt.axis("off")

plt.tight_layout()

plt.show()

print("\n========== TESTING COMPLETED ==========")
