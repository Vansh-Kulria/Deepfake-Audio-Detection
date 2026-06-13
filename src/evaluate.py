import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

def compute_eer(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx], fpr, tpr

def evaluate_tabular(model_data, test_features_path):
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_cols = model_data["feature_cols"]
    
    print(f"Loading test features: {test_features_path}")
    df_test = pd.read_csv(test_features_path)
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    return y_test, y_pred, y_prob

def evaluate_cnn(model_data, test_images_path, test_labels_path):
    import tensorflow as tf
    
    print(f"Loading CNN weights: {model_data['model_path']}")
    model = tf.keras.models.load_model(model_data['model_path'])
    
    print(f"Loading test images: {test_images_path}")
    X_test = np.load(test_images_path)
    y_test = np.load(test_labels_path)
    
    print("Running CNN inference...")
    y_prob = model.predict(X_test, verbose=1).flatten()
    y_pred = (y_prob >= 0.5).astype(np.int32)
    
    return y_test, y_pred, y_prob

def evaluate_model(test_features_path, test_images_path, test_labels_path, model_path="models/detector.pkl", output_dir="results"):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'")
        return
        
    print(f"Loading model metadata: {model_path}")
    model_data = joblib.load(model_path)
    mode = model_data.get("mode", "stats")
    
    if mode == "image":
        if not os.path.exists(test_images_path) or not os.path.exists(test_labels_path):
            print("Error: Missing test images or labels.")
            return
        y_test, y_pred, y_prob = evaluate_cnn(model_data, test_images_path, test_labels_path)
    else:
        if not os.path.exists(test_features_path):
            print("Error: Missing test feature file.")
            return
        y_test, y_pred, y_prob = evaluate_tabular(model_data, test_features_path)
        
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    eer, threshold, fpr, tpr = compute_eer(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    real_acc = tn / (tn + fp)
    fake_acc = tp / (tp + fn)
    
    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy:            {acc * 100:.2f}%")
    print(f"F1-Score:            {f1 * 100:.2f}%")
    print(f"Equal Error Rate:    {eer * 100:.2f}% (Threshold: {threshold:.4f})")
    print(f"Genuine Class Acc:   {real_acc * 100:.2f}%")
    print(f"Synthetic Class Acc: {fake_acc * 100:.2f}%")
    print("--------------------------------")
    print("Confusion Matrix:")
    print(f"  TN: {tn} | FP: {fp}")
    print(f"  FN: {fn} | TP: {tp}")
    print("--------------------------------\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Genuine", "Synthetic"],
        yticklabels=["Genuine", "Synthetic"]
    )
    plt.title("Confusion Matrix Heatmap")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    
    # ROC Curve
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.plot([0, eer], [1 - eer, 1 - eer], color="red", linestyle=":")
    plt.plot([eer, eer], [0, 1 - eer], color="red", linestyle=":")
    plt.scatter([eer], [1 - eer], color="red", s=100, zorder=5, label=f"EER = {eer * 100:.2f}%")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_plot_path, dpi=150)
    plt.close()
    
    print(f"Evaluation plots saved: {output_dir}/")
    
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("AUDIO DEEPFAKE DETECTION PERFORMANCE REPORT\n")
        f.write("===========================================\n")
        f.write(f"Accuracy:            {acc * 100:.2f}%\n")
        f.write(f"F1-Score:            {f1 * 100:.2f}%\n")
        f.write(f"Equal Error Rate:    {eer * 100:.2f}% (Threshold: {threshold:.4f})\n")
        f.write(f"Genuine Class Acc:   {real_acc * 100:.2f}%\n")
        f.write(f"Synthetic Class Acc: {fake_acc * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TN: {tn} | FP: {fp}\n")
        f.write(f"FN: {fn} | TP: {tp}\n")
        
    print(f"Performance report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate audio deepfake detection model performance")
    parser.add_argument("--test_features", type=str, default="data/test_features.csv", help="Path to test features CSV")
    parser.add_argument("--test_images", type=str, default="data/test_images.npy", help="Path to test images .npy")
    parser.add_argument("--test_labels", type=str, default="data/test_labels.npy", help="Path to test labels .npy")
    parser.add_argument("--model", type=str, default="models/detector.pkl", help="Path to model metadata pickle")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save plots and reports")
    
    args = parser.parse_args()
    
    evaluate_model(args.test_features, args.test_images, args.test_labels, args.model, args.output_dir)

if __name__ == "__main__":
    main()
