# Project Completion & Assignment Compliance Audit

> **Audit Date:** 2026-09-28  
> **Repository:** `Y04Assignments/plant-disease-classification`  
> **Project Scope:** Comparative Analysis of Deep Learning Architectures for Plant Disease Classification Using Leaf Images  
> **Auditor:** Antigravity Automated Verification Agent  
> **Verification Standard:** Zero fabrication; strict empirical artifact inspection; code, notebook, results, and Git history ground truth.

---

## 1. Executive Summary

This comprehensive audit evaluates the current completion state, scientific rigor, and university assignment compliance of the project **"Comparative Analysis of Deep Learning Architectures for Plant Disease Classification Using Leaf Images"**.

### Core Findings:
1. **Model Implementation & Training (3 of 4 in Main, 1 Unmerged on Branch):**
   - **Custom CNN (Member 1):** Fully implemented, trained from scratch, evaluated on test data (90.00% accuracy, 0.9819 ROC-AUC), includes a 3-seed stability analysis (90.44% ± 0.77%), error analysis, and Grad-CAM explainability artifacts on `main`.
   - **MobileNetV2 (Member 2):** Fully implemented, trained (baseline + fine-tuning), evaluated on test data (94.67% accuracy, 0.9976 ROC-AUC), error analysis completed, weights saved (`models/mobilenetv2_ft.keras`), and working Streamlit application (`app/app.py`) on `main`. However, its code is housed in `02_Preprocessing.ipynb` while `04_MobileNetV2.ipynb` remains an empty 0-byte file.
   - **ResNet50 (Member 3):** Fully implemented, trained (baseline + Stage 5 fine-tuning), evaluated on test data (98.00% accuracy, 0.9988 ROC-AUC), error analysis completed on `main`. However, the trained weight file (`models/resnet50_leaf_model.keras`) was never committed to the repo, causing `app/app_resnet50.py` to fall back to an artificial heuristic simulation.
   - **EfficientNetB0 (Member 4):** Completed locally by Member 4 on branch `origin/member4/efficientnetb0` (commit `bb0ee61`), achieving 97.33% test accuracy and 0.9966 ROC-AUC, selecting the frozen baseline over fine-tuning. **However, this branch has NOT been merged into `main`, the notebook on `main` is 0 bytes, results/figures were never exported to `results/`, and the model file is not present.**
2. **Dataset & Preprocessing:**
   - The local dataset is intact with 1,532 high-resolution JPEG images across 3 classes (`Healthy`, `Powdery`, `Rust`), 0 corrupt files, and 0 exact duplicates. Dataset provenance and CC0-1.0 license are verified via the Kaggle API.
   - Preprocessing is standardized at 224x224 RGB, but ResNet50 used different augmentation parameters (`RandomFlip("horizontal_and_vertical")` and 0.1 rotation) compared to Custom CNN, MobileNetV2, and EfficientNetB0 (horizontal flip, 0.05 rotation, 0.05 translation).
3. **Modular Code Architecture (`src/`):**
   - The planned reusable Python package under `src/` (`src/preprocessing/`, `src/models/`, `src/training/`) consists solely of empty `__init__.py` files. Most pipeline logic is embedded in interactive Jupyter notebooks.
4. **Major Missing Academic Deliverables:**
   - **Final Academic Report:** Completely MISSING. No draft, LaTeX source, or manuscript exists.
   - **Presentation Slides / Video:** Completely MISSING. No presentation materials or video demo exist.
   - **Viva Preparation:** MISSING. No defense materials exist.
   - **Unified 4-Model Synthesis:** Incomplete. The master comparison table only lists 3 models, and EfficientNetB0 is marked pending.

---

## 2. Project Snapshot

| Property | Verified Value |
| :--- | :--- |
| **Project Title** | Comparative Analysis of Deep Learning Architectures for Plant Disease Classification |
| **Target Problem** | Automated supervised diagnosis of plant leaf fungal pathologies from RGB leaf photos |
| **Classes** | `Healthy`, `Powdery` (Powdery Mildew), `Rust` (Leaf Rust) |
| **Dataset Size on Disk** | 1,532 images (Train: 1,322; Validation: 60; Test: 150) |
| **Architectures Planned** | Custom CNN, MobileNetV2, ResNet50, EfficientNetB0 |
| **Architectures Trained** | 4 of 4 (3 merged on `main`; 1 unmerged on branch `member4/efficientnetb0`) |
| **Best Performing Model** | ResNet50 (Test Accuracy: 98.00%, Weighted F1: 98.00%, ROC-AUC: 0.9988) |
| **Best Edge/Efficiency Model** | MobileNetV2 (2.26M params, 94.67% acc) / Custom CNN (390k params, 90.00% acc) |
| **Interactive Applications** | `app/app.py` (Functional, MobileNetV2); `app/app_resnet50.py` (Heuristic fallback); `app/app_efficientnetb0.py` (Unmerged) |
| **Current Git Branch** | `main` (Head commit: `6eeebec`, tracking `origin/main`) |
| **Total Git Commits** | 15 commits across all branches |

---

## 3. Assignment Requirements Audit

| Req ID | Requirement Description | Evidence Expected | Evidence Found | Status | Evidence File / Path | What Must Be Done Next | Priority |
| :---: | :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| **REQ-01** | Real-world agricultural problem | Problem context, motivation, impact on food security | Thoroughly motivated in `README.md` and notebooks | **COMPLETE** | [README.md](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/README.md#L16-L30) | Synthesize into final report introduction | **LOW** |
| **REQ-02** | Public leaf disease dataset | Downloadable public dataset with clear origin | Kaggle Plant Disease Recognition Dataset | **COMPLETE** | [dataset_metadata.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/dataset_metadata.json) | None | **LOW** |
| **REQ-03** | Dataset justification | Explanation of dataset relevance and domain fit | Detailed in README Section 1 and `01_EDA.ipynb` | **COMPLETE** | [01_EDA.ipynb](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/notebooks/01_EDA.ipynb) | Include in report section 3 | **LOW** |
| **REQ-04** | Dataset provenance | Creator, origin, acquisition date, upload history | Verified via Kaggle API: Rashik Rahman, uploaded 2021-07-04 | **COMPLETE** | [dataset_metadata.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/dataset_metadata.json) | Cite dataset in report references | **LOW** |
| **REQ-05** | Dataset license | Permissive public or academic license verification | Verified CC0-1.0 Public Domain via Kaggle API | **COMPLETE** | [dataset_metadata.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/dataset_metadata.json) | State CC0-1.0 license in report | **LOW** |
| **REQ-06** | Dataset exploration (EDA) | Class distribution, image sizes, color profiles, sample grids | Exhaustive EDA with 8 generated figures and properties CSV | **COMPLETE** | [01_EDA.ipynb](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/notebooks/01_EDA.ipynb), [figures/eda/](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/figures/eda) | Embed EDA figures in report | **LOW** |
| **REQ-07** | Data quality checking | Corrupt files, exact duplicates, near-duplicates, aspect ratio | 0 corrupt, 0 exact duplicates, 2 near-duplicate pairs audited | **COMPLETE** | [eda_summary.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/eda_summary.json), [eda_duplicate_pairs.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/eda_duplicate_pairs.csv) | Discuss data cleaning in report | **LOW** |
| **REQ-08** | Train/Validation/Test split | Stratified partitions with documented ratios | Train: 1322 (86.3%), Val: 60 (3.9%), Test: 150 (9.8%) | **COMPLETE** | [01_EDA.ipynb](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/notebooks/01_EDA.ipynb) | Discuss small validation set in critical analysis | **MEDIUM** |
| **REQ-09** | Test-set isolation | Untouched test split during training and hyperparameter tuning | All notebooks evaluate test split only after freezing weights | **COMPLETE** | Notebooks 02, 03, 05, 06 | Document test isolation protocol in report | **LOW** |
| **REQ-10** | Standardized preprocessing | Resizing, normalization, RGB conversion | Resized to 224x224 RGB; model-specific normalizations verified | **COMPLETE** | [02_Preprocessing.ipynb](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/notebooks/02_Preprocessing.ipynb) | Clean up modular functions in `src/preprocessing/` | **MEDIUM** |
| **REQ-11** | Data augmentation | Training-only augmentation pipeline | RandomFlip, RandomRotation, RandomZoom, RandomTranslation | **PARTIALLY COMPLETE** | Notebooks 02, 03, 05, 06 | Harmonize ResNet50 augmentation discrepancy | **HIGH** |
| **REQ-12** | At least 4 distinct DL architectures | Custom CNN, MobileNetV2, ResNet50, EfficientNetB0 | 3 on `main`; 1 completed on unmerged branch `member4/efficientnetb0` | **PARTIALLY COMPLETE** | `main` + `origin/member4/efficientnetb0` | Merge EfficientNetB0 branch into `main` | **CRITICAL** |
| **REQ-13** | Model training | Convergence curves, loss and accuracy tracking | Training histories logged for all 4 models | **COMPLETE** | [results/](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results) & Branch `06_EfficientNetB0.ipynb` | Export EfficientNetB0 training curves to `results/` | **HIGH** |
| **REQ-14** | Model validation | Validation tracking, early stopping, checkpointing | EarlyStopping & ReduceLROnPlateau utilized across models | **COMPLETE** | Notebooks 02, 03, 05, 06 | None | **LOW** |
| **REQ-15** | Final unseen test evaluation | Quantitative evaluation on 150 test images | Custom CNN (90%), MobileNetV2 (94.67%), ResNet50 (98%), EfficientNetB0 (97.33%) | **PARTIALLY COMPLETE** | Results CSVs + Branch 06 | Merge branch and create unified master table | **CRITICAL** |
| **REQ-16** | Multiple evaluation metrics | Accuracy, Precision, Recall, F1, ROC-AUC | Macro and weighted metrics computed across all models | **COMPLETE** | Results CSVs & Notebook 06 | Integrate EfficientNetB0 into `master_comparison_draft.csv` | **HIGH** |
| **REQ-17** | Confusion matrices | Tabular and visual confusion matrices for each model | CSVs and PNGs exist for CNN, ResNet50; CSV for MNV2; matrix printed in EfficientNetB0 | **PARTIALLY COMPLETE** | [results/](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results) | Generate missing PNGs for MobileNetV2 & EfficientNetB0 | **HIGH** |
| **REQ-18** | Comparative analysis | Side-by-side benchmarking across all 4 architectures | Draft table has 3 models; comparison text not yet written | **PARTIALLY COMPLETE** | [master_comparison_draft.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/master_comparison_draft.csv) | Finalize master table with 4 models and write analysis | **CRITICAL** |
| **REQ-19** | Critical analysis & discussion | Deep examination of tradeoffs, failures, limitations | Qualitative notes exist in README; comprehensive paper missing | **PARTIALLY COMPLETE** | [README.md](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/README.md#L305-L330) | Write critical analysis section in academic report | **HIGH** |
| **REQ-20** | Computational / resource analysis | Params, trainable params, latency, memory, training time | Custom CNN has all metrics; others missing latency & memory | **PARTIALLY COMPLETE** | [master_comparison_draft.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/master_comparison_draft.csv) | Benchmark CPU inference latency for MNV2, ResNet50, EfficientNetB0 | **HIGH** |
| **REQ-21** | Error analysis & failure modes | Detailed misclassification audit, error images, patterns | CNN (15 errors), MNV2 (8 errors), ResNet50 (3 errors) logged; shared errors identified | **PARTIALLY COMPLETE** | [customcnn_error_analysis.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_error_analysis.csv), [mobilenetv2_error_analysis.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_error_analysis.csv), [resnet50_error_analysis.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_error_analysis.csv) | Generate EfficientNetB0 error CSV and cross-model comparison plot | **HIGH** |
| **REQ-22** | Reproducibility | Seeds, pinned dependencies, setup instructions | Fixed seeds (`SEED=42`), `requirements.txt` pinned, reproducible data pipeline | **COMPLETE** | [requirements.txt](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/requirements.txt), [README.md](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/README.md#L735-L770) | Verify end-to-end run from scratch | **MEDIUM** |
| **REQ-23** | Git contribution & history | Meaningful commits, branch workflow, collaboration | 15 commits, 4 contributors, 3 PR merges; spans 11 days | **PARTIALLY COMPLETE** | Git log | Merge PR #4 for Member 4; commit report drafts | **HIGH** |
| **REQ-24** | Academic report | Formal research paper (10 sections) | No report document exists in repository | **MISSING** | None | Write comprehensive academic report | **CRITICAL** |
| **REQ-25** | Presentation / Video demo | Presentation slides or demonstration video | No slides or video artifacts exist in repository | **MISSING** | None | Create slide deck and record presentation | **CRITICAL** |
| **REQ-26** | Viva preparation | Oral examination readiness & architecture mastery | Rule 10 defined in README; no study notes/briefing doc | **MISSING** | None | Compile viva preparation and Q&A briefing sheet | **MEDIUM** |
| **REQ-27** | Working interactive application | Web GUI for inference | `app/app.py` functional; `app_resnet50.py` lacks weights; EfficientNetB0 unmerged | **PARTIALLY COMPLETE** | [app/app.py](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/app/app.py) | Provide unified 4-model selector app or export missing weights | **HIGH** |

---

## 4. Roadmap Audit

Comparing the original project development roadmap (defined in `5fe61e6:README.md` Section 26) against the actual implementation:

| Planned Phase / Milestone | Actual Implementation Status | Status | Evidence | Remaining Work | Priority |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **Phase 1: Foundation**<br>Repo init, venv, gitignore, directory structure, team alignment | Fully implemented on Sep 17, 2026. Gitignore rules, packages, and directory placeholders established. | **COMPLETE** | Commits `39288a3`, `5fe61e6`, `84a9e2a` | None | **LOW** |
| **Phase 2: Dataset & EDA**<br>Dataset download, class balance, image dimensions, quality, duplicates, corruption | Fully executed. Audited 1,532 images. 8 high-resolution figures generated in `figures/eda/`. Verified CC0-1.0 license. | **COMPLETE** | [01_EDA.ipynb](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/notebooks/01_EDA.ipynb), [results/eda_summary.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/eda_summary.json) | None | **LOW** |
| **Phase 3: Preprocessing**<br>Standardized resizing, normalization, data augmentation, leak prevention | Standardized at 224x224 RGB. Training augmentation implemented. Data loaders verified. | **COMPLETE** | [02_Preprocessing.ipynb](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/notebooks/02_Preprocessing.ipynb) | Populate modular files in `src/preprocessing/` | **MEDIUM** |
| **Phase 4: Model Development**<br>Construct Custom CNN, MobileNetV2, ResNet50, EfficientNetB0 | All 4 models developed and trained. CNN, MNV2, ResNet50 on `main`; EfficientNetB0 completed on branch `origin/member4/efficientnetb0`. | **PARTIALLY COMPLETE** | `main` + `origin/member4/efficientnetb0` | Merge EfficientNetB0 branch into `main` | **CRITICAL** |
| **Phase 5: Hyperparameter Tuning**<br>Systematic tuning across LR, batch sizes, dropout, schedules | Custom CNN tuned across 3 configurations (`CNN-EXP-01/02/03`); MNV2 and ResNet50 compared baseline vs fine-tuning. Separate notebook `07_Hyperparameter_Tuning.ipynb` was omitted. | **COMPLETE** | Results CSVs in `results/` | Synthesize tuning comparisons in report | **MEDIUM** |
| **Phase 6: Final Evaluation**<br>Evaluate frozen models on test partition, compute metrics, generate plots | Evaluated across 150 test samples. Detailed metrics logged for all 4 models. Dedicated notebook `08_Final_Evaluation.ipynb` was omitted in favor of per-model notebook evaluation. | **PARTIALLY COMPLETE** | Results CSVs + Branch 06 | Consolidate 4-model evaluation into unified master table | **HIGH** |
| **Phase 7: Critical Analysis**<br>Error analysis, generalization gap, computational tradeoffs, limitations | Error analysis performed for 3 models. Grad-CAM generated for Custom CNN. Master comparison draft partially assembled. Notebook `09_Model_Comparison.ipynb` omitted. | **PARTIALLY COMPLETE** | Results CSVs, [build_master_table.py](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/src/evaluation/build_master_table.py) | Complete cross-architecture synthesis and discussion | **CRITICAL** |
| **Phase 8: Finalization**<br>Academic report, finalize repository, presentation slides/video, viva preparation | None of the finalization documents have been created. | **MISSING** | No report, slide deck, or viva prep files found | Draft academic paper, slide deck, video demo, and viva notes | **CRITICAL** |

---

## 5. Repository Structure Audit

### Directory Inventory:
```text
plant-disease-classification/
├── .agents/                                   # Streamlit skill files
├── .claude/                                   # Streamlit skill files
├── .gitignore                                 # Git exclusion rules
├── README.md                                  # Primary documentation (47 KB)
├── UpdatedReadme.md                           # Alternative/earlier documentation draft (30 KB)
├── requirements.txt                           # Pinned dependencies (148 packages)
├── app/
│   ├── app.py                                 # Streamlit app (loads MobileNetV2)
│   ├── app_resnet50.py                        # Streamlit app (loads ResNet50, heuristic fallback)
│   └── (app_efficientnetb0.py)                # [On branch origin/member4/efficientnetb0 only]
├── dataset/                                   # 1,532 images in Train/, Validation/, Test/ (git-ignored)
├── figures/
│   ├── .gitkeep
│   └── eda/                                   # 8 EDA PNG figures
├── models/
│   ├── .gitkeep
│   └── mobilenetv2_ft.keras                   # Trained MobileNetV2 weights (19.2 MB, git-ignored)
├── notebooks/
│   ├── 01_EDA.ipynb                           # Executed (6.8 MB, 34 cells)
│   ├── 02_Preprocessing.ipynb                 # Executed (5.3 MB, 74 cells - Preprocessing + MobileNetV2)
│   ├── 03_Custom_CNN.ipynb                    # Executed (7.3 MB, 41 cells)
│   ├── 04_MobileNetV2.ipynb                   # EMPTY FILE (0 bytes) - Misplaced in 02_Preprocessing.ipynb
│   ├── 05_ResNet50.ipynb                      # Executed (235 KB, 12 cells)
│   └── 06_EfficientNetB0.ipynb                # EMPTY FILE (0 bytes on main; 38 cells on branch)
├── results/                                   # 26 CSV/JSON/PNG results files
└── src/
    ├── __init__.py                            # Empty
    ├── evaluation/
    │   ├── __init__.py                        # Empty
    │   └── build_master_table.py              # Script to merge result CSVs
    ├── inference/
    │   ├── __init__.py                        # Empty
    │   └── predictor.py                       # Standalone prediction module (MobileNetV2)
    ├── models/
    │   └── __init__.py                        # Empty (planned modular model definitions missing)
    ├── preprocessing/
    │   └── __init__.py                        # Empty (planned modular preprocessing missing)
    └── training/
        └── __init__.py                        # Empty (planned modular training loop missing)
```

### Key Structural Issues:
1. `notebooks/04_MobileNetV2.ipynb` is a **0-byte empty file**. All MobileNetV2 training and evaluation code was placed in `notebooks/02_Preprocessing.ipynb`.
2. `notebooks/06_EfficientNetB0.ipynb` is **0 bytes on `main`**. The 1,660-line executed notebook lives solely on `origin/member4/efficientnetb0`.
3. `src/` modules (`preprocessing`, `models`, `training`) are empty shells with only `__init__.py`. Reusable code was not factored out of notebooks.
4. Model weights are inconsistently distributed: `models/mobilenetv2_ft.keras` exists locally, but `resnet50_leaf_model.keras` and `efficientnetb0_frozen_final.keras` are missing.

---

## 6. Dataset Audit

| Dataset Property | Expected / Claimed | Actual Verified Value | Compliance Status |
| :--- | :--- | :--- | :---: |
| **Source Platform** | Kaggle | Kaggle | **VERIFIED** |
| **Dataset Title** | Plant Disease Recognition Dataset | Plant Disease Recognition Dataset | **VERIFIED** |
| **Kaggle Reference** | `rashikrahmanpritom/plant-disease-recognition-dataset` | `rashikrahmanpritom/plant-disease-recognition-dataset` | **VERIFIED** |
| **Creator** | Rashik Rahman | Rashik Rahman | **VERIFIED** |
| **License** | CC0: Public Domain (CC0-1.0) | CC0: Public Domain (CC0-1.0) | **VERIFIED** |
| **Stated Total Images** | 1,530 images | 1,532 images (+2 discrepancy documented) | **VERIFIED** |
| **Total Images on Disk** | 1,532 images | 1,532 images | **VERIFIED** |
| **Image Format** | JPEG | 1,532 JPEG (100%) | **VERIFIED** |
| **Corrupt Files** | 0 | 0 (all images open and verify successfully) | **VERIFIED** |
| **Exact Duplicates** | 0 | 0 exact byte/MD5 matches | **VERIFIED** |
| **Near-Duplicate Pairs** | Low | 2 pairs identified in training set via dHash | **VERIFIED** |
| **Cross-Split Leakage** | 0 | 0 cross-split duplicates or near-duplicates | **VERIFIED** |
| **Resolution Range** | High resolution | Width: [2421, 5184] px, Height: [1728, 3456] px | **VERIFIED** |
| **Aspect Ratios** | Primarily 4:3 and 3:2 | 1.33 to 1.50 | **VERIFIED** |
| **Number of Classes** | 3 | 3 (`Healthy`, `Powdery`, `Rust`) | **VERIFIED** |
| **Training Split** | ~1,322 images | 1,322 (Healthy: 458, Powdery: 430, Rust: 434) | **VERIFIED** |
| **Validation Split** | 60 images | 60 (Healthy: 20, Powdery: 20, Rust: 20) | **VERIFIED** |
| **Test Split** | 150 images | 150 (Healthy: 50, Powdery: 50, Rust: 50) | **VERIFIED** |
| **Class Balance (Train)**| Near-balanced | Ratio: 1.065 max/min (Healthy: 34.6%, Powdery: 32.5%, Rust: 32.8%) | **VERIFIED** |
| **Class Balance (Val/Test)**| Perfectly balanced | 20 per class (Val), 50 per class (Test) | **VERIFIED** |

### Key Dataset Limitations:
- **Small Validation Split:** The validation partition contains only 60 images (20 per class). A single misclassified image causes a 1.67% drop in validation accuracy. Both MobileNetV2 and ResNet50 achieved 100% validation accuracy, making validation loss the only meaningful differentiator.
- **Limited Environmental Diversity:** Images depict single detached leaves on mostly plain or controlled backgrounds, which does not reflect in-field occlusions, complex canopy shadows, or multiple co-occurring infections.

---

## 7. Preprocessing Audit

| Preprocessing Component | Custom CNN | MobileNetV2 | ResNet50 | EfficientNetB0 | Comparability Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input Resolution** | 224 x 224 x 3 | 224 x 224 x 3 | 224 x 224 x 3 | 224 x 224 x 3 | **Identical** (Standardized) |
| **Color Space** | RGB | RGB | RGB to BGR (Caffe style) | RGB | **Architecture-justified** |
| **Pixel Normalization** | Rescaling `[0, 1]` | Normalization `[-1, 1]` | Mean-subtracted BGR | Built-in `[0, 255]` | **Architecture-justified** (Matches pretraining) |
| **Batch Size** | 32 | 32 | 32 | 32 | **Identical** |
| **Random Seed** | 42 | 42 | 42 | 42 | **Identical** |
| **Horizontal Flip** | Yes | Yes | Yes | Yes | **Identical** |
| **Vertical Flip** | No | No | **Yes** | No | **Discrepancy in ResNet50** |
| **Random Rotation** | 0.05 (±18°) | 0.05 (±18°) | **0.10 (±36°)** | 0.05 (±18°) | **Discrepancy in ResNet50** |
| **Random Translation** | 0.05 | 0.05 | None | 0.05 | **Discrepancy in ResNet50** |
| **Random Zoom** | 0.10 | 0.10 | 0.10 | 0.10 | **Identical** |
| **Prefetching / Cache** | `AUTOTUNE` | `AUTOTUNE` | `AUTOTUNE` | `AUTOTUNE` | **Identical** |
| **Test Preprocessing** | Resize + Scale | Resize + Preprocess | Resize + Preprocess | Resize only | **Architecture-justified** |

> [!NOTE]
> **Fairness Note on Preprocessing:**  
> The differing pixel normalizations (`[-1, 1]` for MobileNetV2, Caffe BGR zero-centering for ResNet50, and raw `[0, 255]` for EfficientNetB0) are strictly required by the respective pretrained ImageNet backbones and represent sound scientific practice. However, the data augmentation pipeline for ResNet50 introduced vertical flipping and double rotation magnitude without translation, diverging from the standardized protocol applied to the other three architectures.

---

## 8. Custom CNN Audit (Member 1)

| Item | Verified Value | Evidence Source |
| :--- | :--- | :--- |
| **Architecture Design** | 4 Convolutional blocks (32, 64, 128, 256 filters), 3x3 kernels, ReLU | `03_Custom_CNN.ipynb`, [customcnn_config.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_config.json) |
| **Regularization** | BatchNormalization after each Conv layer, Dropout (0.3) before Dense head | `03_Custom_CNN.ipynb` |
| **Classification Head** | `GlobalAveragePooling2D()` -> `Dense(3, activation='softmax')` | `03_Custom_CNN.ipynb` |
| **Pretrained Weights** | None (Trained entirely from scratch) | Verified |
| **Total Parameters** | 390,627 | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Trainable Parameters** | 389,667 (960 non-trainable BatchNorm parameters) | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Model Size** | 4.77 MB | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Optimizer & LR** | Adam, Initial LR = 0.001 | [customcnn_config.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_config.json) |
| **Callbacks** | EarlyStopping (patience=10, restore_best_weights=True), ReduceLROnPlateau (factor=0.2, patience=4) | [customcnn_config.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_config.json) |
| **Candidate Experiments** | 3 configs tested: `CNN-EXP-01` (3 blocks), `CNN-EXP-02` (4 blocks), `CNN-EXP-03` (4 blocks + Dense-128) | [customcnn_experiment_comparison.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_experiment_comparison.csv) |
| **Selection Criterion** | Lowest validation loss (`CNN-EXP-02`: val_loss 0.2880, val_acc 90.00%) | [customcnn_config.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_config.json) |
| **Training Duration** | 27 epochs, 1,471.56 seconds (~24.5 minutes on CPU) | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Test Accuracy** | **90.00%** (135 / 150 correct) | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Weighted Precision** | **90.36%** | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Weighted Recall** | **90.00%** | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Weighted F1-Score** | **89.92%** | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Weighted ROC-AUC** | **0.9819** | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Per-Class F1** | Healthy: 0.8868, Powdery: 0.9505, Rust: 0.8602 | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Stability Testing** | 3 seeds (42, 123, 2026): Mean 90.44% ± 0.77% test accuracy | [customcnn_stability_runs.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_stability_runs.csv) |
| **CPU Inference Latency**| 38.03 ms / image | [customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) |
| **Explainability** | Grad-CAM heatmap generated and saved | [results/customcnn_gradcam.png](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_gradcam.png) |
| **Error Analysis** | 15 misclassifications audited and exported | [results/customcnn_error_analysis.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_error_analysis.csv) |

---

## 9. MobileNetV2 Audit (Member 2)

| Item | Verified Value | Evidence Source |
| :--- | :--- | :--- |
| **Backbone & Pretraining**| `tf.keras.applications.MobileNetV2` with ImageNet weights, top excluded | `02_Preprocessing.ipynb` |
| **Classification Head** | `GlobalAveragePooling2D()` -> `Dropout(0.2)` -> `Dense(3, activation='softmax')` | `02_Preprocessing.ipynb` |
| **Baseline Experiment** | `MNV2-BASE-01`: Backbone fully frozen (154 layers frozen), LR = 1e-3, 10 epochs | [mobilenetv2_experiment_comparison.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_experiment_comparison.csv) |
| **Baseline Validation** | Best val_acc: 98.33%, val_loss: 0.0886 (epoch 9), training time: 4.77 min | [mobilenetv2_experiment_comparison.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_experiment_comparison.csv) |
| **Fine-Tuning Experiment**| `MNV2-FT-01`: Unfroze from layer 134 upward (13 trainable backbone layers) | [mobilenetv2_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_final_results.csv) |
| **Fine-Tuning Parameters**| LR = 1e-5, Adam, EarlyStopping (patience=5), best epoch: 5 (stopped at 8) | [mobilenetv2_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_final_results.csv) |
| **Fine-Tuning Validation**| Best val_acc: 100.00%, val_loss: 0.0319, training time: 3.02 min | [mobilenetv2_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_final_results.csv) |
| **Total Parameters** | 2,261,827 | `02_Preprocessing.ipynb` & model file |
| **Trainable Parameters** | 1,198,723 (in fine-tuning stage; 3,843 in baseline stage) | Model file inspection |
| **Model Weight File** | `models/mobilenetv2_ft.keras` exists locally (19.2 MB) | Verified on disk |
| **Test Accuracy** | **94.67%** (142 / 150 correct) | [mobilenetv2_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_final_results.csv) |
| **Weighted Precision** | **94.82%** | [mobilenetv2_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_final_results.csv) |
| **Weighted Recall** | **94.67%** | [mobilenetv2_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_final_results.csv) |
| **Weighted F1-Score** | **94.70%** | [mobilenetv2_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_final_results.csv) |
| **Weighted ROC-AUC** | **0.9976** | [mobilenetv2_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_final_results.csv) |
| **Confusion Matrix** | Healthy: 48 TP, 2 FP Powdery; Powdery: 47 TP, 3 FN Healthy; Rust: 47 TP, 1 FN Healthy, 2 FN Powdery | [mobilenetv2_confusion_matrix.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_confusion_matrix.csv) |
| **Error Analysis** | 8 misclassifications audited and exported | [mobilenetv2_error_analysis.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_error_analysis.csv) |
| **Inference Latency** | Missing from CSV | Not evaluated in notebook |
| **Interactive Demo** | Working Streamlit application loads `mobilenetv2_ft.keras` | [app/app.py](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/app/app.py) |

> [!WARNING]
> **Misplaced Notebook:**  
> All MobileNetV2 code, experiments, evaluation, and logging were implemented inside `notebooks/02_Preprocessing.ipynb`. The assigned notebook `notebooks/04_MobileNetV2.ipynb` remains an empty 0-byte file.

---

## 10. ResNet50 Audit (Member 3)

| Item | Verified Value | Evidence Source |
| :--- | :--- | :--- |
| **Backbone & Pretraining**| `tf.keras.applications.ResNet50` with ImageNet weights, top excluded | `05_ResNet50.ipynb` |
| **Classification Head** | `GlobalAveragePooling2D()` -> `Dropout(0.2)` -> `Dense(3, activation='softmax')` | `05_ResNet50.ipynb` |
| **Baseline Experiment** | `RESNET50-BASE-01`: Backbone fully frozen, LR = 1e-3, 10 epochs, 657.52 s | [resnet50_experiment_comparison.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_experiment_comparison.csv) |
| **Baseline Validation** | Best val_acc: 98.33%, val_loss: 0.0312 (epoch 10) | [resnet50_experiment_comparison.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_experiment_comparison.csv) |
| **Fine-Tuning Experiment**| `RESNET50-FT-01`: Unfroze Stage 5 (`conv5_block1_1_conv` upward) | [resnet50_preprocessing_config.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_preprocessing_config.json) |
| **Fine-Tuning Parameters**| LR = 1e-5, Adam, EarlyStopping (patience=3), 7 epochs trained, best epoch: 4, 505.58 s | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Fine-Tuning Validation**| Best val_acc: 100.00%, val_loss: 0.0040 | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Total Parameters** | 23,593,859 | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Trainable Parameters** | 14,959,619 (Fine-tuning Stage 5 + Head); 6,147 in baseline head | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Total Training Time** | 1,163.10 seconds (~19.4 minutes) | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Test Accuracy** | **98.00%** (147 / 150 correct) | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Weighted Precision** | **98.04%** | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Weighted Recall** | **98.00%** | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Weighted F1-Score** | **98.00%** | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Weighted ROC-AUC** | **0.9988** | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Per-Class F1** | Healthy: 0.9703, Powdery: 0.9796, Rust: 0.9901 | [resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) |
| **Confusion Matrix** | Healthy: 49 TP, 1 FP Rust; Powdery: 48 TP, 2 FN Healthy; Rust: 50 TP, 0 FN | [resnet50_confusion_matrix.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_confusion_matrix.csv) |
| **Error Analysis** | 3 misclassifications (samples 22, 55, 61) audited and exported | [resnet50_error_analysis.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_error_analysis.csv) |
| **Model Weight File** | `models/resnet50_leaf_model.keras` **MISSING from disk** | File not present |
| **Interactive Demo** | `app/app_resnet50.py` falls back to random color heuristic simulation | Code inspection |

> [!CRITICAL]
> **Missing Weight File & Demo Simulation:**  
> The model file `models/resnet50_leaf_model.keras` was generated in Google Colab during notebook execution, but was excluded by `.gitignore` and never committed or transferred to the repository. As a direct consequence, running `app/app_resnet50.py` triggers an `else` branch simulating predictions via RGB pixel averages, rather than executing actual deep learning inference.

---

## 11. EfficientNetB0 Audit (Member 4)

| Item | Verified Value | Evidence Source |
| :--- | :--- | :--- |
| **Branch Location** | Branch `origin/member4/efficientnetb0` (Commit `bb0ee61`) | Git repository |
| **Status on `main`** | **0 bytes (Empty file)** | `notebooks/06_EfficientNetB0.ipynb` on `main` |
| **Pull Request Status** | **Unmerged / No PR created to main** | Git log |
| **Execution State on Branch**| Fully executed (38 total cells, 36 code cells executed) | `origin/member4/efficientnetb0:notebooks/06_EfficientNetB0.ipynb` |
| **Backbone & Pretraining**| `tf.keras.applications.EfficientNetB0` with ImageNet weights, top excluded | Branch notebook Cell 7 |
| **Classification Head** | `GlobalAveragePooling2D()` -> `Dropout(0.2)` -> `Dense(3, activation='softmax')` | Branch notebook Cell 8 |
| **Total Parameters** | 4,053,414 | Branch notebook Cell 9 |
| **Frozen Baseline Run** | 20 epochs max, Adam, LR = 1e-3, EarlyStopping (patience=5) | Branch notebook Cell 10-12 |
| **Baseline Performance** | Best val_acc: 100.00%, val_loss: 0.0540 (epoch 7), training time: 308.29 s | Branch notebook Cell 15, 24 |
| **Fine-Tuning Run** | Unfroze from layer 200 upward (38 trainable backbone layers), LR = 1e-5, 6 epochs | Branch notebook Cell 16-20 |
| **Fine-Tuning Performance**| Best val_acc: 100.00%, val_loss: 0.0607 (epoch 1), training time: 109.26 s | Branch notebook Cell 23, 24 |
| **Candidate Selection Rule**| **Frozen Baseline selected** (Baseline val_loss 0.0540 < Fine-tuning val_loss 0.0607) | Branch notebook Cell 24-25 |
| **Trainable Params (Final)**| 3,843 (Frozen backbone; only classification head trained) | Branch notebook Cell 9, 25 |
| **Test Accuracy** | **97.33%** (146 / 150 correct) | Branch notebook Cell 29, 32, 33 |
| **Weighted Precision** | **97.34%** | Branch notebook Cell 33 |
| **Weighted Recall** | **97.33%** | Branch notebook Cell 33 |
| **Weighted F1-Score** | **97.32%** | Branch notebook Cell 33 |
| **Weighted ROC-AUC** | **0.9966** | Branch notebook Cell 33 |
| **Per-Class Performance** | Healthy: F1 0.9703; Powdery: F1 0.9592; Rust: F1 0.9901 | Branch notebook Cell 32 |
| **Confusion Matrix** | Healthy: 49 TP, 1 FP Powdery; Powdery: 47 TP, 2 FN Healthy, 1 FN Rust; Rust: 50 TP, 0 FN | Branch notebook Cell 34 |
| **Exported Results CSV** | **MISSING** (No CSV files saved to `results/`) | Missing from commit `bb0ee61` |
| **Exported Figures** | **MISSING** (No loss curves or confusion matrix saved to `results/` or `figures/`) | Missing from commit `bb0ee61` |
| **Model Weight File** | Saved locally to Member 4's machine (`D:\Documents\Uni Project Work\...`) | Not present in repo |
| **Streamlit Application** | `app/app_efficientnetb0.py` committed on branch, crashes if model file missing | Branch inspection |

> [!CRITICAL]
> **EfficientNetB0 Gap Scan Summary:**  
> The technical experimentation for EfficientNetB0 is complete and successful (97.33% test accuracy). However, because Member 4 did not merge the branch into `main` and did not export the results CSVs or PNG plots into `results/`, the `main` branch documentation and master comparison table still report EfficientNetB0 as completely missing and unverified.

---

## 12. Experiment & Fairness Audit

To verify that the empirical comparison is scientifically valid and fair:

| Control Dimension | Verification Status | Potential Threat to Validity |
| :--- | :---: | :--- |
| **Dataset Splits** | **Fair** | All 4 models train on identical 1,322 images, validate on identical 60 images, and test on identical 150 images. |
| **Test Set Isolation** | **Fair** | In all 4 models, the test set was evaluated exactly once after candidate model freezing. |
| **Class Index Mapping** | **Fair** | All 4 models map `0: Healthy`, `1: Powdery`, `2: Rust`. |
| **Input Resolution** | **Fair** | All 4 models resize input leaf images to `(224, 224, 3)`. |
| **Evaluation Metrics** | **Fair** | Identical Scikit-learn functions (`classification_report`, `roc_auc_score(multi_class='ovr')`) utilized. |
| **Data Augmentation** | **Slight Discrepancy** | ResNet50 used vertical flipping and 0.10 rotation; CNN, MNV2, and EfficientNetB0 used horizontal flip, 0.05 rotation, 0.05 translation. |
| **Validation Set Granularity**| **Threat** | With only 60 images in validation, accuracy saturates at 100%, forcing selection to rely entirely on tiny validation loss differences. |
| **Fine-Tuning Strategy** | **Methodological Difference** | Custom CNN trained 27 epochs from scratch; MNV2 fine-tuned 13 layers; ResNet50 fine-tuned Stage 5; EfficientNetB0 selected frozen baseline. |

---

## 13. Final Evaluation & Cross-Architecture Benchmark

### Comprehensive Four-Model Comparison Table:

| Metric / Dimension | Custom CNN (Scratch) | MobileNetV2 (Transfer) | ResNet50 (Transfer) | EfficientNetB0 (Transfer) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Paradigm** | Scratch Baseline | Lightweight Inverted Residual | Deep Residual Network | Compound Scaling |
| **Selected Configuration** | 4 Conv blocks (BN + Drop 0.3) | 13 upper layers fine-tuned | Stage 5 residual fine-tuned | Frozen baseline + Dense head |
| **Test Samples** | 150 | 150 | 150 | 150 |
| **Test Accuracy** | 90.00% | 94.67% | **98.00%** | 97.33% |
| **Weighted Precision** | 90.36% | 94.82% | **98.04%** | 97.34% |
| **Weighted Recall** | 90.00% | 94.67% | **98.00%** | 97.33% |
| **Weighted F1-Score** | 89.92% | 94.70% | **98.00%** | 97.32% |
| **Weighted ROC-AUC** | 0.9819 | 0.9976 | **0.9988** | 0.9966 |
| **Healthy F1-Score** | 0.8868 | 0.9412 | **0.9703** | **0.9703** |
| **Powdery F1-Score** | 0.9505 | 0.9400 | **0.9796** | 0.9592 |
| **Rust F1-Score** | 0.8602 | 0.9592 | **0.9901** | **0.9901** |
| **Total Parameters** | **390,627** | 2,261,827 | 23,593,859 | 4,053,414 |
| **Trainable Parameters** | 389,667 | 1,198,723 | 14,959,619 | **3,843** |
| **Total Training Time** | ~1,472 s (~24.5 min) | **~468 s (~7.8 min)** | ~1,163 s (~19.4 min) | ~308 s (~5.1 min) |
| **Model Weight Size** | **4.77 MB** | 19.2 MB | ~214 MB | ~15.5 MB |
| **CPU Latency / Image** | **38.03 ms** | *TBD (Not logged)* | *TBD (Not logged)* | *TBD (Not logged)* |
| **Total Test Errors** | 15 / 150 | 8 / 150 | **3 / 150** | 4 / 150 |

### Scientific Verdict on Research Question:
> *"Which deep-learning architecture provides the best balance between classification performance, generalization, and computational efficiency for plant disease recognition?"*

- **Pure Classification Performance:** **ResNet50** achieves the highest test accuracy (98.00%), weighted F1 (98.00%), and ROC-AUC (0.9988), misclassifying only 3 out of 150 leaves.
- **Optimal Balance of Efficiency & Accuracy:** **EfficientNetB0** (97.33% accuracy with only 3,843 trainable parameters and 4.05M total parameters) and **MobileNetV2** (94.67% accuracy with 2.26M parameters) offer the superior tradeoff for mobile or resource-constrained agricultural edge devices. ResNet50 is nearly 6x larger than EfficientNetB0 and 10x larger than MobileNetV2 for a marginal 0.67% - 3.33% accuracy gain.

---

## 14. Error Analysis Audit

### Misclassification Overview Across 150 Test Images:
- **Custom CNN:** 15 misclassifications (10.0% error rate)
- **MobileNetV2:** 8 misclassifications (5.33% error rate)
- **ResNet50:** 3 misclassifications (2.00% error rate)
- **EfficientNetB0:** 4 misclassifications (2.67% error rate)

### Cross-Model Shared Failure Cases:
1. **Sample Index 55 (`81e5fcf446a9270b.jpg` - True Class: `Powdery`):**
   - Misclassified by **ALL THREE evaluated models** (Custom CNN, MobileNetV2, ResNet50).
   - ResNet50 predicted `Healthy` with ~81% confidence; MobileNetV2 predicted `Healthy` with ~79% confidence.
   - Qualitative inspection reveals an early-stage fungal infection where powdery mildew mycelia are sparse and faint against dark green leaf venation, presenting visual features nearly indistinguishable from healthy leaf texture.
2. **Sample Index 61 (`82c3830f3bd2d1db.jpg` - True Class: `Powdery`):**
   - Misclassified by both **MobileNetV2** and **ResNet50** as `Healthy`.
   - The leaf exhibits localized chlorosis without widespread white sporulation.
3. **Sample Index 22 (`8eb0da4b65b0638f.jpg` - True Class: `Healthy`):**
   - Misclassified by both **Custom CNN** and **ResNet50** as `Rust` with >95% confidence.
   - Natural leaf senescence, minor mechanical leaf bruising, or yellow-brown edge necrotic spots mimic rust pustules.

---

## 15. Reproducibility Audit

| Reproducibility Dimension | Status | Notes / Gaps |
| :--- | :---: | :--- |
| **Deterministic Random Seeds** | **VERIFIED** | `SEED = 42` is fixed in all data loaders and training runs across all notebooks. |
| **Python Virtual Environment** | **VERIFIED** | Local Python 3.12 virtual environment operational in `venv/`. |
| **Dependency Pinning** | **VERIFIED** | `requirements.txt` contains 148 explicitly pinned dependencies (TF 2.21.0, Keras 3.15.1, Streamlit 1.64.0). |
| **Hardware Platform Variance** | **DOCUMENTED** | Member 1 trained on AMD Ryzen CPU (Windows 11); Member 2 on macOS/M-series; Member 3 on Google Colab GPU/CPU; Member 4 on Windows 11. |
| **Data Acquisition Instructions** | **VERIFIED** | README Section 4 documents exact `kagglehub` and Kaggle API commands to reproduce the dataset. |
| **Saved Model Weights** | **PARTIALLY VERIFIED** | Only `mobilenetv2_ft.keras` is preserved in the local repo. Re-evaluating ResNet50 or EfficientNetB0 requires regenerating weights or pulling them from external storage. |
| **Operating System Pathing** | **GAPS DETECTED** | Hardcoded Windows paths (`D:\Documents\...`, `file:///Users/DELL/OneDrive/...`) exist in README and notebook outputs. |

---

## 16. Git & Version Control Audit

### Contribution Summary:
- **Total Commits:** 15 commits
- **Active Branches:** 5 branches (`main`, `feature/preprocessing-mobilenetv2`, `member-1-customcnn`, `member-3-resnet50`, `member4/efficientnetb0`)
- **Pull Requests Merged:** 3 PRs (#1, #2, #3 merged into `main`)
- **Unmerged Pull Requests:** 1 pending (`member4/efficientnetb0` has no open/merged PR)

### Commits by Contributor:
1. **Rushel Ekanayaka / Mona Lisa (Member 2):** 8 commits (Foundation, Preprocessing, MobileNetV2, App, PR merges #1, #3)
2. **Induni Warnachinthaka (Member 3):** 4 commits (ResNet50 implementation, dropout adjustment, path fixes, PR #2)
3. **Tombstone119 / Yohan Kodagoda (Member 1):** 1 commit (Custom CNN runs, results, EDA, PR #3)
4. **Adeesha / zero3nine (Member 4):** 1 commit (EfficientNetB0 notebook and app on unmerged branch)

### Git Hygiene & Security Audit:
- **Large Model Files Tracked:** 0 `.keras` or `.h5` files are tracked in Git history (properly ignored).
- **Dataset Images Tracked:** 0 images tracked in Git history (`dataset/*` properly ignored).
- **API Keys / Secrets Detected:** 0 secrets or credentials detected.
- **Git Commit Cadence:** The entire commit timeline spans 11 days (Sep 17, 2026 to Sep 28, 2026). If the university grading rubric evaluates sustained weekly contributions across a semester, this compressed timeframe may present a rubric vulnerability.

---

## 17. Academic Report Audit

The university assignment requires a formal, comprehensive academic research report.

| Report Section | Required Content | Current Status | Missing Artifacts / Tasks |
| :--- | :--- | :---: | :--- |
| **1. Abstract & Introduction** | Problem statement, agricultural motivation, research questions | **MISSING** | No written draft exists |
| **2. Background & Related Work**| Plant pathology, CNNs, Transfer Learning, MobileNet, ResNet, EfficientNet | **MISSING** | Literature citations and theoretical foundation missing |
| **3. Dataset & EDA** | Provenance, class distribution, resolution, duplicate/leakage audit | **MISSING** | EDA figures exist, but text narrative needs drafting |
| **4. Preprocessing & Augmentation**| Standardized pipeline, color spaces, normalization formulas, leakage guards | **MISSING** | Pipeline exists, write-up missing |
| **5. Experimental Methodology** | Split ratios, seeds, optimizers, callbacks, candidate selection rule | **MISSING** | Experimental protocol needs formalization |
| **6. Model Architectures** | Scratch CNN design, MobileNetV2, ResNet50 Stage 5, EfficientNetB0 | **MISSING** | Architecture diagrams and layer specs missing |
| **7. Results & Comparative Analysis**| Global metrics table, per-class breakdown, learning curves, latency | **MISSING** | Master table needs completion and discussion |
| **8. Critical Analysis & Error Audit**| Shared failure modes, small validation set caveat, domain shift, Grad-CAM | **MISSING** | Error analysis data exists, narrative missing |
| **9. Conclusion & Recommendations** | Final answer to research question, agricultural deployment guidance | **MISSING** | Concluding synthesis missing |
| **10. References** | Peer-reviewed citations (He et al., Sandler et al., Tan & Le, etc.) | **MISSING** | Academic bibliography missing |

---

## 18. Figures & Tables Audit

| Figure / Table Name | Expected Path | Actual File on Disk | Status |
| :--- | :--- | :--- | :---: |
| **Class Distribution Plot** | `figures/eda/class_distribution.png` | `figures/eda/class_distribution.png` (86 KB) | **PRESENT** |
| **Sample Image Grid** | `figures/eda/sample_grid.png` | `figures/eda/sample_grid.png` (3.7 MB) | **PRESENT** |
| **Image Dimension Distribution** | `figures/eda/image_dimensions.png` | `figures/eda/image_dimensions.png` (89 KB) | **PRESENT** |
| **Mean RGB Profiles** | `figures/eda/mean_rgb_by_class.png` | `figures/eda/mean_rgb_by_class.png` (102 KB) | **PRESENT** |
| **Pixel Intensity Distribution** | `figures/eda/pixel_statistics_by_class.png` | `figures/eda/pixel_statistics_by_class.png` (172 KB) | **PRESENT** |
| **Quality Extremes (Dark/Bright)**| `figures/eda/quality_extremes.png` | `figures/eda/quality_extremes.png` (3.5 MB) | **PRESENT** |
| **Near-Duplicate Audit Pairs** | `figures/eda/near_duplicates.png` | `figures/eda/near_duplicates.png` (1.3 MB) | **PRESENT** |
| **Split Distribution Shift** | `figures/eda/split_distribution_shift.png` | `figures/eda/split_distribution_shift.png` (106 KB) | **PRESENT** |
| **Custom CNN Learning Curves** | `results/customcnn_accuracy.png`, `loss.png` | `results/customcnn_accuracy.png`, `loss.png` | **PRESENT** |
| **Custom CNN Confusion Matrix**| `results/customcnn_confusion_matrix.png` | `results/customcnn_confusion_matrix.png` (73 KB) | **PRESENT** |
| **Custom CNN Grad-CAM Heatmap**| `results/customcnn_gradcam.png` | `results/customcnn_gradcam.png` (1.3 MB) | **PRESENT** |
| **MobileNetV2 Baseline Curves** | `results/mobilenetv2_baseline_*.png` | `results/mobilenetv2_baseline_*.png` | **PRESENT** |
| **MobileNetV2 Fine-Tuning Curves**| `results/mobilenetv2_finetuning_*.png`| `results/mobilenetv2_finetuning_*.png` | **PRESENT** |
| **MobileNetV2 Confusion Matrix PNG**| `results/mobilenetv2_confusion_matrix.png` | **MISSING** (Only CSV exists) | **MISSING** |
| **ResNet50 Baseline Curves** | `results/resnet50_baseline_*.png` | `results/resnet50_baseline_*.png` | **PRESENT** |
| **ResNet50 Fine-Tuning Curves** | `results/resnet50_finetuning_*.png` | `results/resnet50_finetuning_*.png` | **PRESENT** |
| **ResNet50 Confusion Matrix PNG**| `results/resnet50_confusion_matrix.png` | `results/resnet50_confusion_matrix.png` (24 KB) | **PRESENT** |
| **EfficientNetB0 Curves & Matrix**| `results/efficientnetb0_*.png` | **MISSING** (Inside notebook outputs on branch only) | **MISSING** |
| **Cross-Model Comparison Bar Charts**| `figures/model_comparison.png` | **MISSING** | **MISSING** |

---

## 19. Streamlit & Demo Application Audit

### Application Inventory:
1. **`app/app.py` (MobileNetV2 Single-Model App):**
   - **Status:** **FULLY FUNCTIONAL**
   - **Model Loaded:** `models/mobilenetv2_ft.keras` (Verified loads cleanly in TF 2.21.0)
   - **Preprocessing:** Resizes to 224x224 RGB, applies `tf.keras.applications.mobilenet_v2.preprocess_input()`
   - **Outputs:** Real-time predicted class, percentage confidence, and progress bars for all 3 class probabilities.
   - **Currently Running:** Active background process verified in IDE metadata.
2. **`app/app_resnet50.py` (ResNet50 App with Fallback):**
   - **Status:** **COMPROMISED / SIMULATION DETECTED**
   - **Defect:** Checks `if HAS_TF and config["path"].exists():` -> `models/resnet50_leaf_model.keras` does NOT exist on disk.
   - **Behavior:** Falls back to an artificial heuristic (`base = np.array([0.96, 0.02, 0.02])`) calculating RGB channel averages rather than executing neural network inference.
3. **`app/app_efficientnetb0.py` (EfficientNetB0 App on Branch):**
   - **Status:** **UNMERGED / WILL CRASH LOCALLY**
   - **Defect:** Raises `FileNotFoundError` immediately if `models/efficientnetb0_frozen_final.keras` does not exist on disk.

---

## 20. Security & Repository Hygiene Audit

| Check | Result | Severity | Details |
| :--- | :---: | :---: | :--- |
| **API Keys & Credentials** | **PASSED** | None | 0 secrets, tokens, or private keys detected in codebase or commit history. |
| **Dataset Tracking in Git** | **PASSED** | None | Raw dataset directory is properly ignored by `.gitignore`. |
| **Model Tracking in Git** | **PASSED** | None | Large `.keras` and `.h5` files are properly excluded from Git tracking. |
| **Hardcoded Personal Absolute Paths** | **FAILED** | **MEDIUM** | Hardcoded Windows OneDrive paths (`file:///Users/DELL/OneDrive/Desktop/...`) found in `README.md` lines 573, 579, and 787. |
| **Windows / macOS Path Inconsistencies** | **WARNING** | **LOW** | Backslashes and Windows drive letters (`D:\Documents\...`) present in Member 4's notebook outputs. |

---

## 21. Documentation Consistency Audit

| Contradiction / Inconsistency | Documented Value | Actual / Verified Value | Source File | Recommended Correction |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet50 Dropout Rate** | `Dropout: 0.3` in Mermaid diagram | `Dropout: 0.2` in text and code | `README.md` L423 vs L415 vs `05_ResNet50.ipynb` | Update Mermaid diagram in `README.md` to `Dropout: 0.2`. |
| **ResNet50 Local File Links** | `file:///Users/DELL/OneDrive/...` | Relative path `models/resnet50_leaf_model.keras` | `README.md` L573, L579, L787 | Replace absolute Windows file URIs with relative repository paths. |
| **MobileNetV2 Notebook File** | Listed as `04_MobileNetV2.ipynb` | 0-byte empty file; code lives in `02_Preprocessing.ipynb` | `notebooks/04_MobileNetV2.ipynb` | Either port MobileNetV2 code into `04_MobileNetV2.ipynb` or update README. |
| **EfficientNetB0 Status** | Marked as "In Progress / Planned" (TBD) | Fully trained and evaluated (97.33% Acc) on branch | `README.md` L649 vs branch `bb0ee61` | Merge branch `origin/member4/efficientnetb0` and update README table. |
| **Master Comparison Table** | 4 models expected | Only 3 models listed; EfficientNetB0 row is blank (`pending`) | [master_comparison_draft.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/master_comparison_draft.csv) | Re-run `build_master_table.py` after populating `efficientnetb0_final_results.csv`. |
| **ResNet50 Model Inference** | Stated to perform real-time DL inference | Falls back to hardcoded color heuristic | [app/app_resnet50.py](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/app/app_resnet50.py) L71 | Provide actual trained weights or note simulation clearly. |
| **Dataset Total Images** | Stated as 1,530 images in Kaggle metadata | Exactly 1,532 images exist on disk (+2 Healthy in Train) | `dataset_metadata.json` vs disk | Keep documented footnote explaining the +2 image discrepancy. |

---

## 22. Completion Percentage Calculation

### Evaluation Methodology:
Weights are allocated across 4 essential university grading pillars:
1. **Technical Implementation (Weight: 40%):** Dataset integrity, preprocessing, scratch CNN, 3 transfer learning models, evaluation pipelines, and interactive demo.
2. **Assignment Compliance (Weight: 25%):** Fulfillment of all 27 explicit criteria, scientific controls, metrics, confusion matrices, and fair comparison protocols.
3. **Documentation & Reporting (Weight: 25%):** Comprehensive academic report (10 sections), README documentation, experiment logging, and literature references.
4. **Presentation & Viva Readiness (Weight: 10%):** Demonstration video, slide deck, and oral examination defense preparation.

```text
1. Technical Implementation:
   - Dataset & EDA: 100%
   - Preprocessing Pipeline: 90%
   - Custom CNN (Member 1): 95%
   - MobileNetV2 (Member 2): 95%
   - ResNet50 (Member 3): 85% (missing weights file)
   - EfficientNetB0 (Member 4): 60% (unmerged branch, missing CSV/plots)
   - Cross-Model Benchmark & Master Table: 45%
   - Reusable Package (src/): 25%
   - Interactive Demo (Streamlit): 70%
   => Technical Implementation Score: 74%

2. Assignment Compliance:
   - Criteria 1-11 (Data & Preprocessing): 98%
   - Criteria 12-16 (4 Models & Evaluation): 85%
   - Criteria 17-21 (Matrices, Errors, Resource Tradeoffs): 75%
   - Criteria 22-23 (Reproducibility & Git): 70%
   - Criteria 24-27 (Report, Presentation, Viva, App): 35%
   => Assignment Compliance Score: 67%

3. Documentation & Reporting:
   - README Documentation: 85%
   - Preserved Metric Logs & Results: 75%
   - Academic Research Report: 0% (Completely missing)
   => Documentation Score: 40%

4. Presentation & Viva Readiness:
   - Presentation Slides: 0%
   - Demonstration Video: 0%
   - Viva Defense Briefing: 0%
   => Presentation & Viva Score: 0%
```

### Calculated Overall Scores:
- **Technical Completion:** **74%**
- **Assignment Compliance:** **67%**
- **Documentation Completion:** **40%**
- **Submission Readiness:** **45%** (Range: 40% - 48%)

---

## 23. What Is Complete

1. **Dataset Ingestion & Provenance Verification:** 1,532 images verified on disk; CC0-1.0 public license confirmed via Kaggle API.
2. **Exploratory Data Analysis (EDA):** Exhaustive audit in `01_EDA.ipynb` with 8 publication-grade figures covering class distributions, resolutions, pixel stats, and duplicate screening.
3. **Data Quality & Leakage Prevention:** 0 corrupt files, 0 cross-split duplicates, strict test-set isolation.
4. **Custom CNN Baseline Pipeline (Member 1):** From-scratch 4-block CNN (`CNN-EXP-02`) achieving 90.00% test accuracy, 3-seed stability analysis, Grad-CAM visualization, and full error audit on `main`.
5. **MobileNetV2 Transfer Learning Pipeline (Member 2):** Baseline (98.33% val acc) and fine-tuning (100% val acc, 94.67% test acc) with complete error audit and working Streamlit app on `main`.
6. **ResNet50 Transfer Learning Pipeline (Member 3):** Baseline and Stage 5 fine-tuning achieving 98.00% test accuracy, 0.9988 ROC-AUC, confusion matrix, and error audit on `main`.
7. **EfficientNetB0 Model Execution (Member 4):** Fully executed in `06_EfficientNetB0.ipynb` on branch `origin/member4/efficientnetb0`, achieving 97.33% test accuracy.
8. **Environment & Dependency Management:** Virtual environment operational with 148 pinned packages in `requirements.txt`.

---

## 24. What Is Partially Complete

1. **Four-Model Master Comparison Table:** `src/evaluation/build_master_table.py` generates `master_comparison_draft.csv`, but EfficientNetB0 is pending and inference latency/model sizes are incomplete.
2. **EfficientNetB0 Artifact Integration:** Notebook executed on branch, but branch is unmerged, results CSVs are not exported to `results/`, and confusion matrix PNG is not saved.
3. **Modular Codebase (`src/`):** Packages exist but contain only empty `__init__.py` files; logic is confined to notebooks.
4. **ResNet50 Demonstration Application:** `app/app_resnet50.py` exists, but executes an artificial color heuristic fallback due to missing local weights.
5. **Confusion Matrix Asset Library:** MobileNetV2 and EfficientNetB0 lack saved standalone confusion matrix PNG plots in `results/`.
6. **Cross-Model Error Analysis:** Shared error cases (e.g., sample 55) identified empirically, but cross-model comparative visualization chart not yet generated.

---

## 25. What Is Missing

1. **Academic Research Report (CRITICAL):** Complete absence of the formal 10-section assignment report.
2. **Presentation Slides (CRITICAL):** No slide deck (.pptx / PDF) summarizing the project, methodology, and results.
3. **Demonstration Video (CRITICAL):** No recorded video walkthrough explaining the research and showing the application.
4. **Viva Preparation Document:** No oral defense briefing sheet or architectural Q&A notes for team members.
5. **Cross-Architecture Comparison Figures:** No grouped bar charts visualizing accuracy, F1-score, parameters, and inference latency across all 4 models.
6. **Dedicated MobileNetV2 Notebook:** `notebooks/04_MobileNetV2.ipynb` is 0 bytes (code is inside `02_Preprocessing.ipynb`).
7. **Trained Model Checkpoints for ResNet50 and EfficientNetB0:** Missing from local `models/` directory.

---

## 26. What Is Unverified

1. **CPU Inference Latency for Pretrained Models:** Latency per image was only measured for Custom CNN (38.03 ms); latency for MobileNetV2, ResNet50, and EfficientNetB0 is unmeasured.
2. **Team Member GitHub Commit Equality:** Member 1 and Member 4 have only 1 commit each in git history, while Member 2 has 8 commits and Member 3 has 4 commits.
3. **Generalization on External Unseen Leaves:** While the models perform strongly on the test partition (90% - 98%), performance on in-field, unsegmented, real-farm smartphone photos remains unverified.

---

## 27. Critical Blockers

The following items directly threaten academic submission or grade outcomes:

1. **MISSING FINAL REPORT:** The written report typically carries 40%–50% of the overall assignment grade. No report currently exists.
2. **MISSING VIDEO / PRESENTATION:** University specifications require a presentation/video demonstration. Neither exists.
3. **UNMERGED EFFICIENTNETB0 BRANCH:** The official repository on `main` only contains 3 completed architectures; the 4th architecture lives on an unmerged branch and `README.md` still reports it as TBD.
4. **FAKE / FALLBACK RESNET50 DEMO:** If an examiner runs `app/app_resnet50.py`, it executes an artificial color heuristic rather than deep learning inference due to missing model weights.

---

## 28. Remaining Work Checklist

### Phase 1: Technical & Repository Consolidation (Immediate Priority)
- [ ] **Task 1.1: Merge Member 4 EfficientNetB0 Branch into `main`**  
  - *Why:* Bring `06_EfficientNetB0.ipynb` and `app/app_efficientnetb0.py` into `main` to fulfill the 4-architecture requirement.  
  - *Folder:* Root / Git branch `member4/efficientnetb0`  
  - *Dependency:* None  
- [ ] **Task 1.2: Export EfficientNetB0 Results to `results/`**  
  - *Why:* Extract `efficientnetb0_final_results.csv`, `confusion_matrix.csv`, and curves from notebook 06 into `results/`.  
  - *Folder:* `results/`  
  - *Dependency:* Task 1.1  
- [ ] **Task 1.3: Update Master Comparison Table**  
  - *Why:* Populate all 4 models in `results/master_comparison_draft.csv` via `src/evaluation/build_master_table.py`.  
  - *Folder:* `src/evaluation/`, `results/`  
  - *Dependency:* Task 1.2  
- [ ] **Task 1.4: Fix ResNet50 Demo Application**  
  - *Why:* Either obtain `resnet50_leaf_model.keras` or provide a unified multi-model selector app in `app/app.py` that gracefully routes to available models.  
  - *Folder:* `app/`  
  - *Dependency:* None  
- [ ] **Task 1.5: Fix Documentation Inconsistencies & Broken Paths**  
  - *Why:* Correct `Dropout: 0.3` Mermaid diagram in `README.md` and replace hardcoded `file:///Users/DELL/...` paths with relative paths.  
  - *Folder:* `README.md`  
  - *Dependency:* None  

### Phase 2: Visualizations & Cross-Model Benchmark
- [ ] **Task 2.1: Benchmark Inference Latency Across All 4 Models**  
  - *Why:* Compute average CPU inference latency (ms) for MNV2, ResNet50, and EfficientNetB0 for fair resource comparison.  
  - *Folder:* `src/evaluation/`  
  - *Dependency:* Task 1.3  
- [ ] **Task 2.2: Generate Unified Comparison Charts**  
  - *Why:* Create high-resolution bar charts comparing Accuracy, F1, Parameters, and Latency for the report.  
  - *Folder:* `figures/`  
  - *Dependency:* Task 2.1  
- [ ] **Task 2.3: Generate Cross-Model Error Comparison Figure**  
  - *Why:* Visualize shared failure cases (Sample 55, 61, 22) across architectures.  
  - *Folder:* `figures/`  
  - *Dependency:* Task 1.2  

### Phase 3: Deliverables & Submission Preparation
- [ ] **Task 3.1: Author Comprehensive Academic Research Report**  
  - *Why:* Complete the mandatory 10-section formal university assignment document.  
  - *Folder:* `report/`  
  - *Dependency:* Phase 1 & Phase 2  
- [ ] **Task 3.2: Create Project Presentation Slide Deck**  
  - *Why:* Slide deck covering motivation, architectures, results, critical analysis, and demo screenshots.  
  - *Folder:* `presentation/`  
  - *Dependency:* Task 3.1  
- [ ] **Task 3.3: Record Demonstration Video**  
  - *Why:* 5-10 minute presentation and live Streamlit inference walkthrough.  
  - *Folder:* Project Root  
  - *Dependency:* Task 3.2  
- [ ] **Task 3.4: Compile Viva Defense Briefing Sheet**  
  - *Why:* Q&A prep document covering depthwise separable convs, residual shortcuts, compound scaling, and loss curves.  
  - *Folder:* `viva/`  
  - *Dependency:* Task 3.1  

---

## 29. Final Submission Readiness Checklist

- [x] Public leaf disease dataset acquired, verified, and audited (1,532 images).
- [x] Exploratory Data Analysis (EDA) completed with 8 saved visual figures.
- [x] Standardized preprocessing and training data augmentation implemented.
- [x] Custom CNN baseline constructed and evaluated from scratch (90.00% Acc).
- [x] MobileNetV2 transfer learning model trained, fine-tuned, and evaluated (94.67% Acc).
- [x] ResNet50 residual model trained, fine-tuned, and evaluated (98.00% Acc).
- [ ] EfficientNetB0 branch merged into `main` and results saved to `results/`.
- [ ] Unified 4-model master comparison table finalized.
- [ ] Comparative benchmark charts generated in `figures/`.
- [ ] Streamlit application updated to cleanly demo models without fallback simulation.
- [ ] Documentation inconsistencies and hardcoded personal paths resolved in `README.md`.
- [ ] Academic Research Report compiled and formatted for submission.
- [ ] Presentation slide deck created.
- [ ] Video demonstration recorded and uploaded/linked.
- [ ] Oral viva preparation briefing completed by all 4 team members.

---

## 30. Evidence & File Index

| Category | File Path | Key Content / Metric |
| :--- | :--- | :--- |
| **Dataset Metadata** | [results/dataset_metadata.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/dataset_metadata.json) | Kaggle origin, Rashik Rahman, CC0-1.0 license |
| **EDA Summary** | [results/eda_summary.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/eda_summary.json) | 1,532 images, class distribution, 0 corrupt files |
| **EDA Visuals** | [figures/eda/](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/figures/eda) | 8 PNG plots (distribution, RGB profiles, duplicates) |
| **Custom CNN Config** | [results/customcnn_config.json](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_config.json) | 4 Conv blocks, BN, Dropout 0.3, Adam LR 0.001 |
| **Custom CNN Results** | [results/customcnn_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_final_results.csv) | 90.00% Acc, 89.92% F1, 0.9819 ROC-AUC, 390k params |
| **Custom CNN Grad-CAM** | [results/customcnn_gradcam.png](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_gradcam.png) | Saliency activation overlay for leaf lesions |
| **Custom CNN Stability** | [results/customcnn_stability_runs.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/customcnn_stability_runs.csv) | 3-seed validation: 90.44% ± 0.77% |
| **MobileNetV2 Results** | [results/mobilenetv2_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_final_results.csv) | 94.67% Acc, 94.70% F1, 0.9976 ROC-AUC |
| **MobileNetV2 Errors** | [results/mobilenetv2_error_analysis.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/mobilenetv2_error_analysis.csv) | 8 misclassified sample indices and probabilities |
| **MobileNetV2 Weights** | [models/mobilenetv2_ft.keras](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/models/mobilenetv2_ft.keras) | Serialized Keras model file (19.2 MB) |
| **ResNet50 Results** | [results/resnet50_final_results.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_final_results.csv) | 98.00% Acc, 98.00% F1, 0.9988 ROC-AUC, 23.6M params |
| **ResNet50 Errors** | [results/resnet50_error_analysis.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_error_analysis.csv) | 3 misclassified sample indices (22, 55, 61) |
| **ResNet50 Matrix** | [results/resnet50_confusion_matrix.png](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/resnet50_confusion_matrix.png) | Visual confusion matrix plot |
| **EfficientNetB0 (Branch)**| `origin/member4/efficientnetb0:06_EfficientNetB0.ipynb` | Executed notebook: 97.33% Acc, 4.05M params |
| **Master Draft Table** | [results/master_comparison_draft.csv](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/results/master_comparison_draft.csv) | Merged metrics for CNN, MNV2, ResNet50 |
| **MobileNetV2 App** | [app/app.py](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/app/app.py) | Verified interactive Streamlit demo |
| **ResNet50 App** | [app/app_resnet50.py](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/app/app_resnet50.py) | Streamlit UI with heuristic fallback logic |
| **Master Table Script** | [src/evaluation/build_master_table.py](file:///Users/gavidurushela/DL%20ass/plant-disease-classification/src/evaluation/build_master_table.py) | Python script compiling result CSVs |

---

# FINAL STATUS

**NOT READY**

Technical completion: 74%  
Assignment compliance: 67%  
Documentation completion: 40%  
Submission readiness: 45%  
