# Sona: Audio Deepfake Detection Pipeline

Sona is a binary classifier that distinguishes genuine human speech from AI-generated (deepfake) audio. It ships with a training pipeline, a CLI tool, and a live Streamlit web dashboard.

Trained and evaluated on the [Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) (for-norm split).

---

## Methodology

### 1. Signal Preprocessing
- All audio is resampled to 16 kHz mono.
- Signals are sliced into non-overlapping 2-second segments (32,000 samples each). Segments shorter than 1 second are discarded; others are zero-padded to reach the target length.
- Silent segments (peak amplitude < 0.01) are skipped to avoid noise bias.

### 2. Mel-Spectrogram Extraction
Each 2-second segment is converted into a 2D time-frequency image:
- 128 Mel frequency bands, FFT window size of 1024, hop length of 256.
- Power values are converted to decibels (dB) relative to peak, then shifted and clipped to the range [-1.0, 1.0].
- Final tensor shape per segment: `128 x 128 x 1`.

### 3. CNN Architecture
A compact convolutional network with Batch Normalization for stable gradient flow:

| Layer | Filters | Output Shape |
|---|---|---|
| Conv2D 3x3, ReLU + BatchNorm | 32 | 128 x 128 x 32 |
| MaxPool 2x2 | — | 64 x 64 x 32 |
| Conv2D 3x3, ReLU + BatchNorm | 64 | 64 x 64 x 64 |
| MaxPool 2x2 | — | 32 x 32 x 64 |
| Conv2D 3x3, ReLU + BatchNorm | 128 | 32 x 32 x 128 |
| MaxPool 2x2 | — | 16 x 16 x 128 |
| Flatten + Dense 128 + Dropout 0.5 | — | 128 |
| Dense 1, Sigmoid | — | 1 |

Optimizer: Adam (lr = 0.0003). Loss: Binary Cross-Entropy.

### 4. Sliding-Window Inference
For recordings longer than 2 seconds, the system slices the full audio into consecutive 2-second windows. Each window is scored independently by the CNN. The final prediction uses the **maximum probability** across all windows — if any segment shows synthetic traits, the whole file is flagged.

### 5. Threshold Calibration
The decision boundary is calibrated using the Equal Error Rate (EER) on a held-out validation split. The EER threshold is the point where the false acceptance rate equals the false rejection rate, giving balanced sensitivity to both classes.

---

## Repository Structure

```
Deepfake-Audio-Detection/
├── data/                    # Extracted .npy arrays and mock samples
├── models/                  # Trained Keras weights and metadata pickle
├── results/                 # Evaluation plots and reports
├── notebooks/
│   └── deepfake_detection.ipynb   # Baseline exploration notebook
├── src/
│   ├── extract_features.py  # Audio-to-spectrogram extraction pipeline
│   ├── train.py             # Model training and threshold computation
│   └── evaluate.py          # Test-set evaluation and chart generation
├── app.py                   # Streamlit web application
├── predict.py               # CLI single-file prediction tool
├── download_dataset.py      # Kaggle dataset download helper
├── requirements.txt         # Python dependencies
├── packages.txt             # System-level dependencies for deployment
└── README.md
```

---

## Quick Start

### 1. Setup
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python download_dataset.py
```

### 3. Extract Features
```bash
# Training split (10,000 files per class)
python src/extract_features.py --data_dir "PATH_TO_DATASET" --split training --mode image --max_files 10000

# Testing split
python src/extract_features.py --data_dir "PATH_TO_DATASET" --split testing --mode image --max_files 5000
```

### 4. Train
```bash
python src/train.py --train_images data/training_images.npy --train_labels data/training_labels.npy --epochs 20 --batch_size 64 --model cnn
```

### 5. Evaluate
```bash
python src/evaluate.py --test_images data/testing_images.npy --test_labels data/testing_labels.npy --model models/detector.pkl
```

### 6. Run Predictions
```bash
# CLI
python predict.py --file "path/to/audio.wav"

# Web dashboard
streamlit run app.py
```

---

## Dataset

**The Fake-or-Real Dataset** (Kaggle)
- Training: ~27,000 genuine + ~27,000 synthetic files
- Validation: ~5,400 per class
- Testing: ~2,300 per class
- Audio format: 16 kHz mono WAV, normalized amplitude
- Split used: `for-norm/for-norm/`

---

## Limitations

- The model is trained on traditional TTS/vocoder-based deepfakes. Modern neural vocoders (ElevenLabs, Bark, etc.) produce waveforms that may not trigger detection.
- Audio with heavy background noise, music, or non-speech content may produce unreliable predictions.
- Very short clips (< 1 second of speech) provide limited spectral information for classification.

---

## License

This project is built for academic and research purposes.
