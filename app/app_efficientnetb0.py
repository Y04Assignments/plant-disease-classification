import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# ==========================================
# Configuration
# ==========================================

CLASS_NAMES = [
    "Healthy",
    "Powdery",
    "Rust"
]

IMG_SIZE = (224, 224)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "efficientnetb0_frozen_final.keras"
)


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Plant Disease Recognition - EfficientNetB0",
    page_icon="🌿",
    layout="centered"
)


# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ==========================================
# Prediction Function
# ==========================================

def predict_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to EfficientNetB0 input size
    image = image.resize(IMG_SIZE)

    # Convert to NumPy
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # EfficientNetB0 preprocessing
    #
    # Keras EfficientNetB0 includes its preprocessing
    # inside the model, so do NOT use:
    # tf.keras.applications.mobilenet_v2.preprocess_input()
    #
    # The model receives pixel values in the 0-255 range.

    # Prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ==========================================
# UI
# ==========================================

st.title("🌿 Plant Disease Recognition")

st.write(
    "Upload a plant leaf image to classify it as "
    "Healthy, Powdery, or Rust using EfficientNetB0."
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Plant Leaf",
        use_container_width=True
    )

    st.divider()

    if st.button(
        "🔍 Detect Disease",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing the leaf image..."
        ):

            (
                predicted_class,
                confidence,
                probabilities
            ) = predict_image(image)

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        st.subheader("Class Probabilities")

        for i, class_name in enumerate(CLASS_NAMES):

            st.write(
                f"**{class_name}:** "
                f"{probabilities[i]:.2%}"
            )

            st.progress(
                float(probabilities[i])
            )

        st.divider()

        st.caption(
            "This is an experimental deep-learning "
            "classification system. Results should be "
            "interpreted as model predictions."
        )