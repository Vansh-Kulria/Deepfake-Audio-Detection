# Sona: Audio Deepfake Detection

Sona is a machine learning system designed to classify speech recordings as either **Genuine (Human)** or **Synthetic (AI-Generated)**. 

Our pipeline extracts acoustic spectral and cepstral features (MFCCs and LFCCs) and trains a robust ensemble classifier (XGBoost) to identify the acoustic signatures of generative speech vocoders and voice-cloning engines.

---

## 🚀 Key Deliverables Included
1. **Interactive Streamlit Web App (`app.py`)**: A web interface to upload WAV files, view predictions, inspect confidence scores, play audio, and see waveform visualizations.
2. **Analysis Notebook (`notebooks/deepfake_detection.ipynb`)**: A step-by-step walk-through of feature extraction, exploratory visualization of genuine vs. fake speech, model training, and metric checks.
3. **Training & Inference scripts**:
   - `src/extract_features.py`: Batch extraction of MFCCs/LFCCs and statistical aggregates.
   - `src/train.py`: Standardizes features, splits data, and trains XGBoost/Random Forest.
   - `src/evaluate.py`: Tests saved models and saves confusion matrices and ROC curves.
   - `predict.py`: Command-line interface to classify a single audio file.
4. **Preconfigured environment**: Virtual environment configuration (`.venv`) and `.gitignore` safety locks.

---

## 📊 Evaluation Criteria & Thresholds
To be verified, the model must achieve or exceed the following performance on the evaluation dataset:

| Metric | Target Threshold | Description |
|---|---|---|
| **Overall Accuracy** | **$\ge$ 80%** | Fraction of correctly identified clips overall. |
| **Equal Error Rate (EER)** | **$\le$ 12%** | Balance point between False Acceptance and False Rejection. |
| **F1 Score** | **$\ge$ 80%** | Harmonic mean of precision and recall. |
| **Per-Class Accuracy** | **$\ge$ 75%** | Both classes (Genuine & Deepfake) individually. |

---

## 🛠️ Feature Extraction Methodology

Deepfake audio synthesis models (like Google Wavenet, Tacotron, or voice cloners) leave artifacts in the high-frequency spectrum and spectral envelopes. To capture these anomalies, we extract a rich combination of features:
- **Mel-Frequency Cepstral Coefficients (MFCCs)**: 20 coefficients capturing the overall shape of the vocal tract filter.
- **Linear Frequency Cepstral Coefficients (LFCCs)**: 20 coefficients. Unlike MFCCs (which scale logarithmically mimicking human hearing), LFCCs scale linearly, retaining higher frequency detail where synthetic vocoder noise is often present.
- **Deltas ($\Delta$) & Delta-Deltas ($\Delta-\Delta$)**: Temporal derivatives capturing rate of change in acoustic spectral parameters.
- **Spectral Centroid, Spectral Rolloff, & Zero Crossing Rate (ZCR)**: Captures spectral shape, brightness, and high-frequency noise profiles.

For each sequence of frame-level coefficients, we compute four statistical moments to summarize the clip into a static feature vector:
1. **Mean** (spectral center of mass)
2. **Standard Deviation** (spectral dynamics/fluctuations)
3. **Skewness** (asymmetry of spectral shapes)
4. **Kurtosis** (presence of extreme transients/spikes)

This results in a total of **~488 tabular features** per audio file.

---

## 📁 Recommended Project Layout
The repository is structured as follows:
```text
Deepfake-Audio-Detection/
├── data/                    # Local cache directory for dataset files (git-ignored)
├── models/                  # Saved model pickles (git-ignored)
├── notebooks/
│   └── deepfake_detection.ipynb  # Comprehensive narrative analysis
├── results/                 # Evaluation reports, ROC, and confusion matrix plots
├── src/
│   ├── extract_features.py  # Audio feature extraction pipeline
│   ├── train.py             # Preprocessing & training pipeline
│   └── evaluate.py          # Detailed evaluation suite
├── .gitignore               # Ignores large datasets and binary models
├── app.py                   # Streamlit interactive application
├── predict.py               # CLI single-file predictor
├── download_dataset.py      # Automated script to fetch the Kaggle dataset
└── requirements.txt         # Project python dependencies
```

---

## 💻 How to Run the Pipeline

### 1. Prerequisites and Setup
Ensure you have Python 3.12+ installed. Set up the local virtual environment and activate it:
```bash
# Set up virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Download the Dataset
We use **The Fake-or-Real Dataset** from Kaggle. Run our automated downloader:
```bash
python download_dataset.py
```
This script downloads the 16 GB dataset via `kagglehub` into your user cache. Note down the path it prints (e.g., `C:\Users\AS\.cache\kagglehub\datasets\mohammedabdeldayem\the-fake-or-real-dataset\versions\2`).

### 3. Extract Features
Point the feature extractor to the downloaded dataset directory. We extract training and testing features (capping to 5,000 files per class for training speed, which yields a balanced dataset of 10,000 files):
```bash
# Extract training features
python src/extract_features.py --data_dir "PATH_TO_DOWNLOADED_DATASET" --split train --max_files 5000

# Extract test features (for validation checks)
python src/extract_features.py --data_dir "PATH_TO_DOWNLOADED_DATASET" --split test --max_files 1000
```
This outputs `data/train_features.csv` and `data/test_features.csv`.

### 4. Train the Model
Train the classifier (XGBoost is the default):
```bash
python src/train.py --train_features data/train_features.csv
```
This saves the trained model package (including scaler and feature lists) to `models/detector.pkl`.

### 5. Evaluate the Model
Evaluate on the independent test split and generate performance heatmaps:
```bash
python src/evaluate.py --test_features data/test_features.csv
```
This prints the evaluation report and saves `confusion_matrix.png` and `roc_curve.png` in `results/`.

### 6. Command-Line Inference
Test any raw `.wav` voice file using the CLI script:
```bash
python predict.py --file "path/to/test_file.wav"
```

### 7. Run the Deployed Web App
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the web application.
