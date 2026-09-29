import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


# ==========================================
# Configuration
# ==========================================

CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

IMG_SIZE = (224, 224)

PROJECT_ROOT = Path(__file__).resolve().parent

# If app.py is inside the /app folder, move to project root
if not (PROJECT_ROOT / "models").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

COMPARISON_PATH = (
    RESULTS_DIR / "master_comparison_draft.csv"
)


# ==========================================
# Model Configuration
# ==========================================

MODEL_CONFIGS = {

    "EfficientNetB0": {
        "path": MODELS_DIR / "efficientnetb0_frozen_final.keras",
        #assuming the model includes preprocessing, set to "none"
        "preprocessing": "none", 
    },

    "MobileNetV2": {
        "path": MODELS_DIR / "mobilenetv2_ft.keras",
        "preprocessing": "none",
    },

    "ResNet50": {
        "path": MODELS_DIR / "resnet50_leaf_model.keras",
        "preprocessing": "none",
    },

    "Custom CNN": {
        # change to CNN model path
        "path": MODELS_DIR / "efficientnetb0_frozen_final.keras",
        "preprocessing": "none",
    },
}


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="centered"
)


# ==========================================
# Sidebar - Model Selection
# ==========================================

st.sidebar.title("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    options=list(MODEL_CONFIGS.keys()),
    index=0,
    help="Select which trained model to use for image classification."
)

selected_config = MODEL_CONFIGS[selected_model_name]

st.sidebar.markdown(
    f"**Model file:** `{selected_config['path'].name}`"
)


# ==========================================
# Model Loading
# ==========================================

@st.cache_resource
def load_model(model_path):

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return tf.keras.models.load_model(model_path)


# ==========================================
# Prediction Function
# ==========================================

def predict_image(
    image,
    model,
    preprocessing
):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to model input size
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

    # ------------------------------------------
    # Model-specific preprocessing (if any)
    # ------------------------------------------

    if preprocessing == "none":
        processed = image_array

    else:
        raise ValueError(
            f"Unknown preprocessing type: {preprocessing}"
        )

    # Prediction
    probabilities = model.predict(
        processed,
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
# Main UI
# ==========================================

st.title("🌿 Plant Disease Recognition")

st.write(
    "Select a deep-learning model and upload a plant "
    "leaf image to classify it as Healthy, Powdery, or Rust."
)

st.divider()

tab_prediction, tab_comparison = st.tabs([
    "Prediction",
    "Model Comparison"
])

# ==========================================
# Prediction Tab
# ==========================================

with tab_prediction:

    st.subheader(f"Prediction using {selected_model_name}")

    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Uploaded Image")
        st.image(image, caption="Plant Leaf", use_container_width=True)

        st.divider()

        if st.button("🔍 Detect Disease", use_container_width=True):
            try:
                with st.spinner(
                    f"Analyzing with {selected_model_name}..."
                ):
                    model = load_model(str(selected_config["path"]))
                    (
                        predicted_class,
                        confidence,
                        probabilities
                    ) = predict_image(
                        image,
                        model,
                        selected_config["preprocessing"]
                    )

                st.success(
                    f"Prediction: {predicted_class}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2%}"
                )

                st.subheader(
                    "Class Probabilities"
                )

                for i, class_name in enumerate(
                    CLASS_NAMES
                ):
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

            except FileNotFoundError as e:
                st.error(str(e))

            except Exception as e:
                st.error(f"Unable to run the selected model: {e}")


# ==========================================
# Model Comparison Tab
# ==========================================

with tab_comparison:
    st.subheader("Model Comparison")

    st.write(
        "Comparison of the evaluated plant-disease "
        "classification models."
    )

    st.divider()

    if not COMPARISON_PATH.exists():
        st.error("Master comparison file was not found.")
        st.caption(f"Expected location: {COMPARISON_PATH}")

    else:
        try:
            comparison_df = pd.read_csv(COMPARISON_PATH)

            st.dataframe(comparison_df, use_container_width=True,hide_index=True)

        except Exception as e:
            st.error(f"Unable to load the comparison file: {e}")