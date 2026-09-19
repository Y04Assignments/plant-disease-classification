import numpy as np
import tensorflow as tf
from PIL import Image


CLASS_NAMES = [
    "Healthy",
    "Powdery",
    "Rust"
]

IMG_SIZE = (224, 224)

MODEL_PATH = "models/mobilenetv2_ft.keras"


# Load model once
model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image: Image.Image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(IMG_SIZE)

    # Convert image to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(probabilities[predicted_index])

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
    }