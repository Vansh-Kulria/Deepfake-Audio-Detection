import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve

def compute_eer(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]

def train_tabular_model(train_features_path, val_features_path, model_type):
    df_train = pd.read_csv(train_features_path)
    cols_to_drop = ["label", "file_name"]
    feature_cols = [c for c in df_train.columns if c not in cols_to_drop]
    
    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    
    if val_features_path and os.path.exists(val_features_path):
        df_val = pd.read_csv(val_features_path)
        X_val = df_val[feature_cols].values
        y_val = df_val["label"].values
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if model_type == "xgboost":
        import xgboost as xgb
        print("Training XGBoost Classifier...")
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss"
        )
        model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    else:
        print("Training Random Forest Classifier...")
        model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
    y_val_pred = model.predict(X_val_scaled)
    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
    
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    eer, threshold = compute_eer(y_val, y_val_prob)
    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = cm.ravel()
    real_acc = tn / (tn + fp)
    fake_acc = tp / (tp + fn)
    
    print_results(acc, f1, eer, threshold, real_acc, fake_acc, tn, fp, fn, tp)
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/detector.pkl"
    joblib.dump({
        "mode": "stats",
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "model_type": model_type,
        "threshold": float(threshold),
        "val_metrics": {
            "accuracy": acc, "f1": f1, "eer": eer,
            "genuine_acc": real_acc, "deepfake_acc": fake_acc, "cm": cm.tolist()
        }
    }, model_path)
    print(f"Model saved: {model_path}")

def train_cnn_model(train_images_path, train_labels_path, val_images_path, val_labels_path, epochs, batch_size):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    X_train = np.load(train_images_path)
    y_train = np.load(train_labels_path)
    
    if val_images_path and os.path.exists(val_images_path):
        X_val = np.load(val_images_path)
        y_val = np.load(val_labels_path)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
    
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training Convolutional Neural Network...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    y_val_prob = model.predict(X_val, verbose=0).flatten()
    y_val_pred = (y_val_prob >= 0.5).astype(np.int32)
    
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    eer, threshold = compute_eer(y_val, y_val_prob)
    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = cm.ravel()
    real_acc = tn / (tn + fp)
    fake_acc = tp / (tp + fn)
    
    print_results(acc, f1, eer, threshold, real_acc, fake_acc, tn, fp, fn, tp)
    
    os.makedirs("models", exist_ok=True)
    cnn_path = "models/detector.keras"
    meta_path = "models/detector.pkl"
    
    model.save(cnn_path)
    joblib.dump({
        "mode": "image",
        "model_path": cnn_path,
        "model_type": "cnn",
        "threshold": float(threshold),
        "val_metrics": {
            "accuracy": acc, "f1": f1, "eer": eer,
            "genuine_acc": real_acc, "deepfake_acc": fake_acc, "cm": cm.tolist()
        }
    }, meta_path)
    
    print(f"CNN weights saved: {cnn_path}")
    print(f"Model package saved: {meta_path}")

def print_results(acc, f1, eer, threshold, real_acc, fake_acc, tn, fp, fn, tp):
    print("\n--- Validation Performance Metrics ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"EER:       {eer * 100:.2f}% (Threshold: {threshold:.4f})")
    print(f"Genuine Class Acc:   {real_acc * 100:.2f}%")
    print(f"Synthetic Class Acc: {fake_acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  TN: {tn} | FP: {fp}")
    print(f"  FN: {fn} | TP: {tp}")
    print("--------------------------------------\n")
    
    print("Threshold Validation Checks:")
    print(f"  Accuracy >= 80%:      {'Passed' if acc >= 0.8 else 'Failed'}")
    print(f"  EER <= 12%:           {'Passed' if eer <= 0.12 else 'Failed'}")
    print(f"  F1 Score >= 80%:      {'Passed' if f1 >= 0.8 else 'Failed'}")
    print(f"  Genuine Acc >= 75%:   {'Passed' if real_acc >= 0.75 else 'Failed'}")
    print(f"  Synthetic Acc >= 75%: {'Passed' if fake_acc >= 0.75 else 'Failed'}\n")

def main():
    parser = argparse.ArgumentParser(description="Train classification models for audio deepfake detection")
    parser.add_argument("--train_features", type=str, default="data/train_features.csv", help="Path to training CSV features")
    parser.add_argument("--val_features", type=str, default=None, help="Path to validation CSV features")
    parser.add_argument("--train_images", type=str, default="data/train_images.npy", help="Path to training images .npy")
    parser.add_argument("--train_labels", type=str, default="data/train_labels.npy", help="Path to training labels .npy")
    parser.add_argument("--val_images", type=str, default=None, help="Path to validation images .npy")
    parser.add_argument("--val_labels", type=str, default=None, help="Path to validation labels .npy")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train CNN")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to train CNN")
    parser.add_argument("--model", type=str, default="cnn", choices=["xgboost", "random_forest", "cnn"], help="Classifier type to train")
    
    args = parser.parse_args()
    
    if args.model == "cnn":
        if not os.path.exists(args.train_images) or not os.path.exists(args.train_labels):
            print(f"Error: Missing image inputs. Extract features in 'image' mode first.")
            return
        train_cnn_model(args.train_images, args.train_labels, args.val_images, args.val_labels, args.epochs, args.batch_size)
    else:
        if not os.path.exists(args.train_features):
            print(f"Error: Missing tabular feature file: {args.train_features}")
            return
        train_tabular_model(args.train_features, args.val_features, args.model)

if __name__ == "__main__":
    main()
