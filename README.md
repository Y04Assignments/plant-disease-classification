# Plant Disease Recognition Using Deep Learning

A university project for **supervised image classification of plant leaf diseases**. The system classifies leaf images into three classes: **Healthy, Powdery Mildew, and Rust**. The project compares four CNN architectures—**Custom CNN, MobileNetV2, ResNet50, and EfficientNetB0**—to investigate differences in classification performance, generalization, and computational efficiency.

The dataset contains **1,532 locally verified images**, split into 1,322 training, 60 validation, and 150 test images. Images are resized to **224×224** for model input.

**Dataset:** [Kaggle Plant Disease Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset?utm_source=chatgpt.com)

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Y04Assignments/plant-disease-classification.git
cd plant-disease-classification
```

### 2. Create and activate the virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Download the dataset from the Kaggle link above and place it inside:

```text
dataset/
├── Train/
├── Validation/
└── Test/
```

The project expects the three classes:

```text
Healthy/
Powdery/
Rust/
```

The dataset is excluded from Git through `.gitignore`.

### 5. Run the Streamlit application

From the project root:

```bash
python -m streamlit run app/app.py
```

The application allows users to upload a leaf image and receive a predicted disease class, confidence score, and class probability distribution.

### 6. Run the notebooks

The model experiments are located in:

```text
notebooks/
├── 01_Custom_CNN.ipynb
├── 02_MobileNetV2.ipynb
├── 03_ResNet50.ipynb
└── 04_EfficientNetB0.ipynb
```

These cover dataset analysis, preprocessing, model training, fine-tuning, and evaluation.
