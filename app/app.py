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
    / "mobilenetv2_ft.keras"
)


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="centered"
)


# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ==========================================
# Prediction Function
# ==========================================

def predict_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to MobileNetV2 input size
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

    # MobileNetV2 preprocessing
    image_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(image_array)
    )

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
    "Healthy, Powdery, or Rust."
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

            predicted_class, confidence, probabilities = (
                predict_image(image)
            )

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