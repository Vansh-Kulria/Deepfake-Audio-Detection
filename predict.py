import os
import argparse
import joblib
import numpy as np
import pandas as pd
import librosa
import scipy.stats as stats

def extract_stats(y, sr):
    features = {}
    
    def add_moments(matrix, prefix):
        for idx in range(matrix.shape[0]):
            row = matrix[idx, :]
            features[f"{prefix}_{idx}_mean"] = float(np.mean(row))
            features[f"{prefix}_{idx}_std"] = float(np.std(row))
            features[f"{prefix}_{idx}_skew"] = float(stats.skew(row))
            features[f"{prefix}_{idx}_kurt"] = float(stats.kurtosis(row))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    add_moments(mfcc, "mfcc")
    add_moments(mfcc_delta, "mfcc_delta")
    add_moments(mfcc_delta2, "mfcc_delta2")
    
    try:
        lfcc = librosa.feature.lfcc(y=y, sr=sr, n_lfcc=20)
        lfcc_delta = librosa.feature.delta(lfcc)
        lfcc_delta2 = librosa.feature.delta(lfcc, order=2)
        add_moments(lfcc, "lfcc")
        add_moments(lfcc_delta, "lfcc_delta")
        add_moments(lfcc_delta2, "lfcc_delta2")
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
    
    return features

def extract_spectrogram(file_path):
    try:
        target_sr = 16000
        target_len = target_sr * 2
        
        y, sr = librosa.load(file_path, sr=target_sr)
        if len(y) == 0:
            return None
            
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]
            
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        if S_db.shape[1] < 128:
            diff = 128 - S_db.shape[1]
            S_db = np.pad(S_db, ((0, 0), (0, diff)), mode='constant')
        elif S_db.shape[1] > 128:
            S_db = S_db[:, :128]
            
        S_db = (S_db + 40.0) / 40.0
        S_db = np.clip(S_db, -1.0, 1.0)
        
        return S_db
    except Exception:
        return None

def predict_audio(file_path, model_path="models/detector.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at '{file_path}'")
        
    model_data = joblib.load(model_path)
    mode = model_data.get("mode", "stats")
    
    if mode == "image":
        import tensorflow as tf
        
        cnn_model_path = model_data["model_path"]
        if not os.path.exists(cnn_model_path):
            raise FileNotFoundError(f"CNN model weights not found at '{cnn_model_path}'")
            
        model = tf.keras.models.load_model(cnn_model_path)
        img = extract_spectrogram(file_path)
        if img is None:
            raise ValueError("Could not extract Mel spectrogram from audio")
            
        X = np.expand_dims(img, axis=0)
        X = np.expand_dims(X, axis=-1)
        
        prob = model.predict(X, verbose=0).flatten()[0]
        
        threshold = model_data.get("threshold", 0.5)
        if prob >= threshold:
            pred_class = "Synthetic (AI-Generated)"
            confidence = prob
        else:
            pred_class = "Genuine (Human)"
            confidence = 1.0 - prob
            
        return pred_class, confidence
    else:
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_cols = model_data["feature_cols"]
        
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) == 0:
            raise ValueError("Could not load audio signal")
            
        feat_dict = extract_stats(y, sr)
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
        
        class_map = {0: "Genuine (Human)", 1: "Synthetic (AI-Generated)"}
        return class_map[pred_label], confidence

def main():
    parser = argparse.ArgumentParser(description="Predict if a wav file is Genuine (Human) or Synthetic (AI-Generated)")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to the audio .wav file")
    parser.add_argument("--model", "-m", type=str, default="models/detector.pkl", help="Path to the model metadata (.pkl)")
    
    args = parser.parse_args()
    
    try:
        pred_class, confidence = predict_audio(args.file, args.model)
        print(f"\nResult: {pred_class} (Confidence: {confidence * 100:.2f}%)")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
