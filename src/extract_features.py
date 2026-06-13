import os
import glob
import argparse
import random
import numpy as np
import pandas as pd
import librosa
import scipy.stats as stats
from tqdm import tqdm

def get_audio_files(base_path, split_name="train"):
    real_files = []
    fake_files = []
    
    possible_paths = [
        os.path.join(base_path, split_name),
        os.path.join(base_path, "for-norm", split_name),
        os.path.join(base_path, "for-norm", "for-norm", split_name),
        os.path.join(base_path, "for-2sec", split_name),
        os.path.join(base_path, "for-original", split_name)
    ]
    
    found_split = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "real")) or os.path.exists(os.path.join(p, "fake")):
            found_split = p
            break
            
    if found_split:
        real_dir = os.path.join(found_split, "real")
        fake_dir = os.path.join(found_split, "fake")
        
        if os.path.exists(real_dir):
            real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        if os.path.exists(fake_dir):
            fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))
            
        print(f"Directory located: {found_split}")
        print(f"  Genuine files:  {len(real_files)}")
        print(f"  Synthetic files: {len(fake_files)}")
        return real_files, fake_files
        
    print(f"Split directory '{split_name}' not found at base. Walking path recursively...")
    for root, dirs, files in os.walk(base_path):
        if not split_name or split_name in root.lower():
            if os.path.basename(root) == "real":
                real_files.extend([os.path.join(root, f) for f in files if f.endswith(".wav")])
            elif os.path.basename(root) == "fake":
                fake_files.extend([os.path.join(root, f) for f in files if f.endswith(".wav")])
                
    if len(real_files) == 0 and len(fake_files) == 0 and split_name:
        print("No files matched the split filter. Re-scanning recursively for all files...")
        for root, dirs, files in os.walk(base_path):
            if os.path.basename(root) == "real":
                real_files.extend([os.path.join(root, f) for f in files if f.endswith(".wav")])
            elif os.path.basename(root) == "fake":
                fake_files.extend([os.path.join(root, f) for f in files if f.endswith(".wav")])

    print(f"Files found:")
    print(f"  Genuine:   {len(real_files)}")
    print(f"  Synthetic: {len(fake_files)}")
    return real_files, fake_files

def extract_stats(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) == 0:
            return None
            
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
        except AttributeError:
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
    except Exception:
        return None

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

def main():
    parser = argparse.ArgumentParser(description="Extract features for Option A (stats) or Option B (image)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save extracted features")
    parser.add_argument("--max_files", type=int, default=5000, help="Maximum number of files per class to process")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to extract: train, validation, or test")
    parser.add_argument("--mode", type=str, default="image", choices=["stats", "image"], help="Feature mode: stats (XGBoost) or image (CNN)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    real_files, fake_files = get_audio_files(args.data_dir, args.split)
    
    if len(real_files) == 0 and len(fake_files) == 0:
        print("Error: No audio files found. Verify dataset path.")
        return
        
    if args.max_files > 0:
        if len(real_files) > args.max_files:
            random.seed(42)
            real_files = random.sample(real_files, args.max_files)
        if len(fake_files) > args.max_files:
            random.seed(42)
            fake_files = random.sample(fake_files, args.max_files)
            
    images_dataset = []
    labels_dataset = []
    stats_dataset = []
    
    print(f"Extracting features (Mode: {args.mode})...")
    for f in tqdm(real_files, desc="Genuine"):
        if args.mode == "image":
            img = extract_spectrogram(f)
            if img is not None:
                images_dataset.append(img)
                labels_dataset.append(0)
        else:
            feat = extract_stats(f)
            if feat is not None:
                feat["label"] = 0
                feat["file_name"] = os.path.basename(f)
                stats_dataset.append(feat)
                
    for f in tqdm(fake_files, desc="Synthetic"):
        if args.mode == "image":
            img = extract_spectrogram(f)
            if img is not None:
                images_dataset.append(img)
                labels_dataset.append(1)
        else:
            feat = extract_stats(f)
            if feat is not None:
                feat["label"] = 1
                feat["file_name"] = os.path.basename(f)
                stats_dataset.append(feat)
                
    if args.mode == "image":
        X = np.array(images_dataset, dtype=np.float32)
        X = np.expand_dims(X, axis=-1)
        y = np.array(labels_dataset, dtype=np.int32)
        
        out_img = os.path.join(args.output_dir, f"{args.split}_images.npy")
        out_lbl = os.path.join(args.output_dir, f"{args.split}_labels.npy")
        
        np.save(out_img, X)
        np.save(out_lbl, y)
        print(f"Features saved: {out_img} & {out_lbl} (Shape: {X.shape})")
    else:
        df = pd.DataFrame(stats_dataset)
        out_csv = os.path.join(args.output_dir, f"{args.split}_features.csv")
        df.to_csv(out_csv, index=False)
        print(f"Features saved: {out_csv} (Shape: {df.shape})")

if __name__ == "__main__":
    main()
