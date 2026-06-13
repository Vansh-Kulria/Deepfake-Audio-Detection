import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sona — Audio Authenticator",
    page_icon="🔈",
    layout="centered"
)

st.markdown("""
<style>
    .block-container {
        max-width: 620px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    .header-container {
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 24px;
    }
    .logo {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--text-color);
        text-transform: lowercase;
    }
    .tagline {
        font-size: 13px;
        color: var(--text-color);
        opacity: 0.6;
        margin-top: 2px;
    }
    .result-box {
        border-radius: 6px;
        padding: 18px 22px;
        margin-top: 20px;
        margin-bottom: 24px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        background-color: rgba(128, 128, 128, 0.04);
    }
    .result-box.authentic {
        border-left: 4px solid #10b981;
    }
    .result-box.synthetic {
        border-left: 4px solid #ef4444;
    }
    .result-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.2px;
        color: var(--text-color);
        opacity: 0.5;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .result-status {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 4px;
    }
    .result-confidence {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-color);
    }
    .result-desc {
        font-size: 13px;
        color: var(--text-color);
        opacity: 0.7;
        line-height: 1.5;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_metadata(model_path="models/detector.pkl"):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_neural_network(weights_path):
    import tensorflow as tf
    if os.path.exists(weights_path):
        return tf.keras.models.load_model(weights_path)
    return None

def extract_spectrogram(file_path):
    try:
        y_full, sr = librosa.load(file_path, sr=16000)
        if len(y_full) == 0:
            return None, None, 0
            
        duration = librosa.get_duration(y=y_full, sr=sr)
        segment_len = 16000 * 2
        
        segments = []
        if len(y_full) <= segment_len:
            padded = np.pad(y_full, (0, segment_len - len(y_full)), mode='constant')
            segments.append(padded)
        else:
            num_segments = len(y_full) // segment_len
            for idx in range(num_segments):
                segments.append(y_full[idx * segment_len : (idx + 1) * segment_len])
                
            remainder = len(y_full) % segment_len
            if remainder > 16000:
                trailing = y_full[-remainder:]
                padded = np.pad(trailing, (0, segment_len - len(trailing)), mode='constant')
                segments.append(padded)
                
        segments = segments[:15]
        
        spectrograms = []
        for seg in segments:
            if np.max(np.abs(seg)) < 0.01:
                continue
                
            S = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            if S_db.shape[1] < 128:
                diff = 128 - S_db.shape[1]
                S_db = np.pad(S_db, ((0, 0), (0, diff)), mode='constant')
            else:
                S_db = S_db[:, :128]
                
            S_db = (S_db + 40.0) / 40.0
            S_db = np.clip(S_db, -1.0, 1.0)
            spectrograms.append(S_db)
            
        if not spectrograms:
            seg = y_full[:segment_len]
            if len(seg) < segment_len:
                seg = np.pad(seg, (0, segment_len - len(seg)), mode='constant')
            S = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
            S_db = librosa.power_to_db(S, ref=np.max)
            if S_db.shape[1] < 128:
                diff = 128 - S_db.shape[1]
                S_db = np.pad(S_db, ((0, 0), (0, diff)), mode='constant')
            else:
                S_db = S_db[:, :128]
            S_db = (S_db + 40.0) / 40.0
            S_db = np.clip(S_db, -1.0, 1.0)
            spectrograms.append(S_db)
            
        return spectrograms, y_full, duration
    except Exception:
        return None, None, 0

def extract_features_stats(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) == 0:
            return None, None, 0
            
        features = {}
        import scipy.stats as stats
        
        def compute_moments(matrix, prefix):
            for idx in range(matrix.shape[0]):
                row = matrix[idx, :]
                features[f"{prefix}_{idx}_mean"] = float(np.mean(row))
                features[f"{prefix}_{idx}_std"] = float(np.std(row))
                features[f"{prefix}_{idx}_skew"] = float(stats.skew(row))
                features[f"{prefix}_{idx}_kurt"] = float(stats.kurtosis(row))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        compute_moments(mfcc, "mfcc")
        compute_moments(mfcc_delta, "mfcc_delta")
        compute_moments(mfcc_delta2, "mfcc_delta2")
        
        try:
            lfcc = librosa.feature.lfcc(y=y, sr=sr, n_lfcc=20)
            lfcc_delta = librosa.feature.delta(lfcc)
            lfcc_delta2 = librosa.feature.delta(lfcc, order=2)
            compute_moments(lfcc, "lfcc")
            compute_moments(lfcc_delta, "lfcc_delta")
            compute_moments(lfcc_delta2, "lfcc_delta2")
        except Exception:
            pass

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        features["spec_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spec_centroid_std"] = float(np.std(spectral_centroid))
        features["spec_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spec_rolloff_std"] = float(np.std(spectral_rolloff))
        features["zcr_mean"] = float(np.mean(zero_crossing_rate))
        features["zcr_std"] = float(np.std(zero_crossing_rate))
        
        duration = librosa.get_duration(y=y, sr=sr)
        return features, y, duration
    except Exception:
        return None, None, 0

# Header Brand Identity
st.markdown("""
<div class="header-container">
    <div class="logo">sona</div>
    <div class="tagline">Audio Authentication and Classification Pipeline</div>
</div>
""", unsafe_allow_html=True)

model_data = load_metadata()

audio_path = None
is_sample = False

uploaded_file = st.file_uploader("Upload audio recording (WAV format)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name
else:
    mock_fake = "data/mock/fake/deepfake_0.wav"
    mock_real = "data/mock/real/genuine_0.wav"
    if os.path.exists(mock_fake) and os.path.exists(mock_real):
        sample_choice = st.selectbox(
            "Or select a sample file to test:",
            ["None", "Mock Synthetic (Deepfake)", "Mock Authentic (Genuine)"]
        )
        if sample_choice == "Mock Synthetic (Deepfake)":
            audio_path = mock_fake
            is_sample = True
        elif sample_choice == "Mock Authentic (Genuine)":
            audio_path = mock_real
            is_sample = True
        if is_sample:
            st.audio(audio_path, format="audio/wav")

if audio_path is not None:
    with st.spinner("Analyzing audio sample..."):
        if model_data and model_data.get("mode", "stats") == "image":
            img_list, y, duration = extract_spectrogram(audio_path)
            feat_dict = None
        else:
            feat_dict, y, duration = extract_features_stats(audio_path)
            img_list = None
            
    if uploaded_file is not None:
        try:
            os.unlink(audio_path)
        except Exception:
            pass
            
    if img_list is not None or feat_dict is not None:
        if model_data is None:
            st.error("Classification model not initialized. Please train the model first.")
        else:
            mode = model_data.get("mode", "stats")
            pred_label = None
            
            if mode == "image":
                cnn_model = load_neural_network(model_data["model_path"])
                if cnn_model is not None:
                    probs = []
                    for img in img_list:
                        X = np.expand_dims(img, axis=0)
                        X = np.expand_dims(X, axis=-1)
                        prob = cnn_model.predict(X, verbose=0).flatten()[0]
                        probs.append(prob)
                    
                    prob = max(probs) if probs else 0.0
                    
                    threshold = model_data.get("threshold", 0.5)
                    if prob >= threshold:
                        pred_label = 1
                        confidence = prob
                    else:
                        pred_label = 0
                        confidence = 1.0 - prob
                else:
                    st.error("Could not load neural network weights.")
            else:
                model = model_data["model"]
                scaler = model_data["scaler"]
                feature_cols = model_data["feature_cols"]
                
                df_single = pd.DataFrame([feat_dict])
                for col in feature_cols:
                    if col not in df_single.columns:
                        df_single[col] = 0.0
                X = df_single[feature_cols].values
                X_scaled = scaler.transform(X)
                
                threshold = model_data.get("threshold", 0.5)
                probs = model.predict_proba(X_scaled)[0]
                prob_synthetic = probs[1]
                if prob_synthetic >= threshold:
                    pred_label = 1
                    confidence = prob_synthetic
                else:
                    pred_label = 0
                    confidence = 1.0 - prob_synthetic
                
            if pred_label is not None:
                if is_sample:
                    st.caption(f"Analyzing sample: **{os.path.basename(audio_path)}**")
                    
                if pred_label == 0:
                    st.markdown(f"""
                    <div class="result-box authentic">
                        <div class="result-label">Classification</div>
                        <div class="result-status">Authentic Voice</div>
                        <div class="result-confidence">Confidence Score: {confidence * 100:.2f}%</div>
                        <div class="result-desc">The acoustic properties of this sample are consistent with genuine, natural human speech. No anomalous vocal signatures were detected.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box synthetic">
                        <div class="result-label">Classification</div>
                        <div class="result-status">Synthetic / AI-Generated</div>
                        <div class="result-confidence">Confidence Score: {confidence * 100:.2f}%</div>
                        <div class="result-desc">Acoustic markers show patterns characteristic of digital voice synthesis or vocoder artifacts. High probability of deepfake audio.</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### Waveform Analysis")
                fig, ax = plt.subplots(figsize=(8, 1.8), facecolor='none')
                ax.plot(np.linspace(0, duration, len(y)), y, color='#475569', alpha=0.9, lw=0.5)
                ax.set_facecolor('none')
                ax.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                
                st.markdown("#### Sample Metadata")
                meta_col1, meta_col2 = st.columns(2)
                with meta_col1:
                    st.caption(f"Duration: {duration:.2f}s")
                with meta_col2:
                    st.caption(f"Sample Rate: 16 kHz")

st.markdown("---")

with st.expander("Verification pipeline details"):
    st.markdown("**1. Feature Representation**")
    st.write("Converts input signals into normalized representations (either multi-moment cepstral matrices or scaled Mel-spectrogram tensors).")
    st.markdown("**2. Binary Classifier**")
    st.write("Applies the trained model to identify temporal anomalies and boundary inconsistencies left by voice synthesizers.")
