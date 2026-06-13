# Sona: Audio Deepfake Detection Pipeline

Sona is a machine learning system designed to classify speech recordings as either genuine (human) or synthetic (AI-generated). This repository contains the feature extraction pipeline, deep learning models, training and evaluation scripts, and a web dashboard for inference.

---

## Methodology

### 1. Signal Preprocessing
To ensure consistency across raw audio signals of varying formats:
- **Resampling**: All incoming WAV audio is downsampled to 16 kHz monophonic channel representation.
- **Temporal Alignment**: Signal segments are padded or truncated to a fixed duration of 2.0 seconds (32,000 samples). Padding uses constant zero-value padding to prevent edge transients.

### 2. Time-Frequency Representation
Audio signals are transformed into 2D time-frequency representations to capture spectral boundary traces:
- **Mel-Spectrogram**: Computed using a Hann window with a Fast Fourier Transform (FFT) size of 1024, a hop length of 256 samples, and 128 Mel frequency bands. This maps 2.0 seconds of audio to a spectral grid of shape `(128, 126)`.
- **Dimensional Alignment**: The time dimension is right-padded with zero columns to match the frequency dimension, producing a square grid of shape `128x128`.
- **Normalization**: Power levels are converted to decibels (dB) relative to maximum peak power. Decibel scales are shifted and normalized into a uniform $[-1.0, 1.0]$ range suitable for deep neural network input weights.

### 3. Classification Architecture
The system uses a lightweight 2D Convolutional Neural Network (CNN) optimized for fast CPU evaluation while maintaining boundary detection accuracy:
- **Feature Extraction Layer**: Three sequential 2D convolutional layers with small kernel windows ($3\times3$, ReLU activations) and 16, 32, and 64 filters respectively. Each block is downscaled using a $2\times2$ Max Pooling filter.
- **Classification Head**: Flattened features map to a Dense layer (64 hidden units, ReLU) with a Dropout rate of 0.5 to mitigate overfitting, concluding with a single Sigmoid activation neuron.

### 4. Decision Threshold Calibration
Standard classifiers use a generic decision threshold of 0.5. However, acoustic vocoders produce varied distributions. Sona computes the **Equal Error Rate (EER)** boundary on the validation split:
- **Equal Error Rate (EER)**: The cutoff where False Acceptance Rate (FAR) equals False Rejection Rate (FRR).
- **Inference Alignment**: The calculated EER threshold is saved to model metadata. Single-file predictions and dashboard checks use this calibrated threshold to balance per-class sensitivity.

---

## Repository Structure

```text
Deepfake-Audio-Detection/
├── data/                    # NumPy training and testing arrays (git-ignored)
├── models/                  # Package pickles and Keras weights (git-ignored)
├── results/                 # Plot charts and evaluation summaries
├── src/
│   ├── extract_features.py  # Audio preprocessing and Mel-spectrogram tensor extraction
│   ├── train.py             # Model training, EER calculation, and serialization
│   └── evaluate.py          # Test split validation and metrics chart plotting
├── app.py                   # Streamlit web application
├── predict.py               # CLI single-file prediction utility
├── download_dataset.py      # Script to automate downloading dataset files from Kaggle
├── requirements.txt         # Package dependencies
└── README.md                # System documentation
```

---

## Execution Pipeline

### 1. Environment Configuration
Ensure Python 3.12+ is installed, then set up the local environment:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Ingestion
Download the source dataset (*The Fake-or-Real Dataset* from Kaggle):
```bash
python download_dataset.py
```
Note the destination folder path printed by the script (e.g. `C:\Users\AS\.cache\kagglehub\datasets\mohammedabdeldayem\the-fake-or-real-dataset\versions\2`).

### 3. Audio Preprocessing & Feature Extraction
Process the raw WAV directory structures and output normalized spectrogram grids:
```bash
# Process training files (caps at 5,000 files per class)
python src/extract_features.py --data_dir "PATH_TO_DATASET" --split training --mode image --max_files 5000

# Process testing files (caps at 1,000 files per class)
python src/extract_features.py --data_dir "PATH_TO_DATASET" --split testing --mode image --max_files 1000
```
This saves `training_images.npy`, `training_labels.npy`, `testing_images.npy`, and `testing_labels.npy` under the `data/` folder.

### 4. Neural Network Training
Train the CNN model weights and compute decision thresholds:
```bash
python src/train.py --train_images data/training_images.npy --train_labels data/training_labels.npy --epochs 10 --batch_size 64 --model cnn
```
Outputs are serialized to `models/detector.keras` (network weights) and `models/detector.pkl` (calibrated EER thresholds and scaler configurations).

### 5. Final Evaluation
Evaluate generalizability on the test set and generate performance charts:
```bash
python src/evaluate.py --test_images data/testing_images.npy --test_labels data/testing_labels.npy --model models/detector.pkl
```
Prints accuracy reports and saves `confusion_matrix.png` and `roc_curve.png` inside the `results/` folder.

### 6. Command-Line Inference
Verify any standalone WAV recording using the prediction CLI:
```bash
python predict.py --file "path/to/recording.wav"
```

### 7. Interactive Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```

---

## Model Metrics

Model performance evaluated on the independent test split (2,000 files) yields:

- **Overall Accuracy**: 83.75%
- **F1-Score**: 82.35%
- **Validation Split EER**: 0.50% (Decision Threshold: 0.7786)
- **Test Set EER**: 15.40%
- **Genuine Class Accuracy**: 91.70%
- **Synthetic Class Accuracy**: 75.80%
