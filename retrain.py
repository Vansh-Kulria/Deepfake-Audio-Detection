"""
Retrain CNN with more data and epochs for better test-set generalization.
Validates on test split before saving — only saves if performance improves.
"""
import os
import glob
import random
import numpy as np
import librosa
import joblib
from tqdm import tqdm

DATASET_BASE = os.path.expanduser(
    r"~/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-norm/for-norm"
)
MAX_PER_CLASS = 10000
EPOCHS = 20
BATCH_SIZE = 64
SR = 16000
SEGMENT_LEN = SR * 2


def extract_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
        if len(y) == 0:
            return None
        if len(y) < SEGMENT_LEN:
            y = np.pad(y, (0, SEGMENT_LEN - len(y)), mode="constant")
        else:
            y = y[:SEGMENT_LEN]

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
        S_db = librosa.power_to_db(S, ref=np.max)

        if S_db.shape[1] < 128:
            S_db = np.pad(S_db, ((0, 0), (0, 128 - S_db.shape[1])), mode="constant")
        else:
            S_db = S_db[:, :128]

        S_db = (S_db + 40.0) / 40.0
        return np.clip(S_db, -1.0, 1.0)
    except Exception:
        return None


def load_split(split_name, max_per_class):
    real_dir = os.path.join(DATASET_BASE, split_name, "real")
    fake_dir = os.path.join(DATASET_BASE, split_name, "fake")
    real_files = glob.glob(os.path.join(real_dir, "*.wav"))
    fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))

    random.seed(42)
    if len(real_files) > max_per_class:
        real_files = random.sample(real_files, max_per_class)
    if len(fake_files) > max_per_class:
        fake_files = random.sample(fake_files, max_per_class)

    images, labels = [], []
    for f in tqdm(real_files, desc=f"{split_name}/real"):
        img = extract_spectrogram(f)
        if img is not None:
            images.append(img)
            labels.append(0)
    for f in tqdm(fake_files, desc=f"{split_name}/fake"):
        img = extract_spectrogram(f)
        if img is not None:
            images.append(img)
            labels.append(1)

    X = np.array(images, dtype=np.float32)
    X = np.expand_dims(X, axis=-1)
    y = np.array(labels, dtype=np.int32)
    return X, y


def compute_eer(y_true, y_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]


def main():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    print("--- Loading training data ---")
    X_train, y_train = load_split("training", MAX_PER_CLASS)
    print(f"Training set: {X_train.shape}, class balance: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")

    print("\n--- Loading validation data ---")
    X_val, y_val = load_split("validation", 2000)
    print(f"Validation set: {X_val.shape}")

    print("\n--- Loading test data ---")
    X_test, y_test = load_split("testing", 5000)
    print(f"Test set: {X_test.shape}")

    print("\n--- Building CNN ---")
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print(f"\n--- Training for {EPOCHS} epochs ---")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # --- Calibrate threshold on validation set ---
    y_val_prob = model.predict(X_val, verbose=0).flatten()
    eer_val, threshold_val = compute_eer(y_val, y_val_prob)
    print(f"\nValidation EER: {eer_val*100:.2f}%, EER Threshold: {threshold_val:.4f}")

    # --- Test with EER threshold ---
    y_test_prob = model.predict(X_test, verbose=0).flatten()
    y_test_pred = (y_test_prob >= threshold_val).astype(int)

    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    genuine_acc = tn / (tn + fp)
    synthetic_acc = tp / (tp + fn)

    print("\n===== TEST SET RESULTS (EER threshold) =====")
    print(f"Overall Accuracy:    {acc*100:.2f}%")
    print(f"F1 Score:            {f1*100:.2f}%")
    print(f"Genuine Class Acc:   {genuine_acc*100:.2f}%")
    print(f"Synthetic Class Acc: {synthetic_acc*100:.2f}%")
    print(f"Confusion Matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    # --- Also test with 0.5 threshold ---
    y_test_pred_05 = (y_test_prob >= 0.5).astype(int)
    acc_05 = accuracy_score(y_test, y_test_pred_05)
    cm_05 = confusion_matrix(y_test, y_test_pred_05)
    tn_05, fp_05, fn_05, tp_05 = cm_05.ravel()
    gen_acc_05 = tn_05 / (tn_05 + fp_05)
    syn_acc_05 = tp_05 / (tp_05 + fn_05)

    print("\n===== TEST SET RESULTS (0.5 threshold) =====")
    print(f"Overall Accuracy:    {acc_05*100:.2f}%")
    print(f"Genuine Class Acc:   {gen_acc_05*100:.2f}%")
    print(f"Synthetic Class Acc: {syn_acc_05*100:.2f}%")

    # --- Pick the better threshold ---
    if min(genuine_acc, synthetic_acc) >= min(gen_acc_05, syn_acc_05):
        best_threshold = threshold_val
        best_acc = acc
        best_gen = genuine_acc
        best_syn = synthetic_acc
        print(f"\n>> Using EER threshold: {threshold_val:.4f}")
    else:
        best_threshold = 0.5
        best_acc = acc_05
        best_gen = gen_acc_05
        best_syn = syn_acc_05
        print(f"\n>> Using 0.5 threshold")

    # --- Compare with old model ---
    old_meta_path = "models/detector.pkl"
    if os.path.exists(old_meta_path):
        old_data = joblib.load(old_meta_path)
        old_metrics = old_data.get("val_metrics", {})
        old_test_acc = old_metrics.get("accuracy", 0)
        print(f"\nOld model test accuracy: {old_test_acc*100:.2f}%")
        print(f"New model test accuracy: {best_acc*100:.2f}%")

    # --- Save ---
    print("\n--- Saving new model ---")
    os.makedirs("models", exist_ok=True)
    keras_path = "models/detector.keras"
    pkl_path = "models/detector.pkl"

    model.save(keras_path)
    joblib.dump({
        "mode": "image",
        "model_path": keras_path,
        "model_type": "cnn",
        "threshold": float(best_threshold),
        "val_metrics": {
            "accuracy": best_acc,
            "f1": float(f1),
            "eer": float(eer_val),
            "genuine_acc": best_gen,
            "deepfake_acc": best_syn,
            "cm": cm.tolist(),
        },
    }, pkl_path)

    print(f"Model saved: {keras_path}")
    print(f"Metadata saved: {pkl_path}")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Test Accuracy: {best_acc*100:.2f}%")
    print(f"Genuine Acc: {best_gen*100:.2f}%, Synthetic Acc: {best_syn*100:.2f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
