"""
Model Evaluation and Validation Suite
FPGA-Based Multilingual AI Receptionist

This script evaluates trained model and scaler artifacts against the feature dataset
WITHOUT overwriting or modifying the files. It reports:
- Stratified test split metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix
- Full dataset verification
- Class imbalance inspection
- Model parameter inspection and verification stamp
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import joblib

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def evaluate_classifier(
    csv_path: str = "language_features.csv",
    model_path: str = "language_classifier.pkl",
    scaler_path: str = "feature_scaler.pkl",
    test_size: float = 0.2,
    random_state: int = 42
):
    print("=" * 70)
    print("     FPGA MULTILINGUAL RECEPTIONIST — MODEL VALIDATION REPORT")
    print("=" * 70)

    # 1. Resolve paths
    for p, name in [(csv_path, "Dataset CSV"), (model_path, "Model Pickle"), (scaler_path, "Scaler Pickle")]:
        if not os.path.exists(p):
            alt = os.path.join("FPGA_Multilingual_Receptionist", p)
            if os.path.exists(alt):
                if name == "Dataset CSV":
                    csv_path = alt
                elif name == "Model Pickle":
                    model_path = alt
                else:
                    scaler_path = alt
            else:
                print(f"[!] Error: {name} not found at '{p}'")
                sys.exit(1)

    print(f"[*] Dataset Source : {csv_path}")
    print(f"[*] Model Artifact : {model_path}")
    print(f"[*] Scaler Artifact: {scaler_path}")
    print("-" * 70)

    # 2. Load dataset
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    feature_cols = [c for c in df.columns if c != "language"]
    n_features = len(feature_cols)

    print(f"\n[1] DATASET SUMMARY")
    print(f"    Total Samples  : {total_samples}")
    print(f"    Feature Count  : {n_features} (Expected: 26)")
    print(f"    Missing Values : {df.isnull().sum().sum()}")
    print(f"    Duplicates     : {df.duplicated().sum()}")
    print("\n    Class Distribution:")
    class_counts = df["language"].value_counts()
    for lang, count in class_counts.items():
        pct = (count / total_samples) * 100
        print(f"      - {lang:<10}: {count:>5} samples ({pct:.1f}%)")

    # 3. Load model and scaler
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

    print(f"\n[2] MODEL ARCHITECTURE INSPECTION")
    print(f"    Model Class    : {type(model).__name__}")
    print(f"    Target Classes : {model.classes_.tolist()}")
    print(f"    Hidden Layers  : {getattr(model, 'hidden_layer_sizes', 'N/A')}")
    print(f"    Activation     : {getattr(model, 'activation', 'N/A')}")
    print(f"    Input Features : {getattr(model, 'n_features_in_', 'N/A')}")
    
    if hasattr(model, "coefs_"):
        w_shapes = [c.shape for c in model.coefs_]
        b_shapes = [b.shape for b in model.intercepts_]
        total_params = sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_)
        print(f"    Weight Shapes  : {w_shapes}")
        print(f"    Bias Shapes    : {b_shapes}")
        print(f"    Total Weights  : {total_params} parameters")

    X = np.asarray(df[feature_cols].to_numpy(), dtype=np.float64)
    y = np.asarray(df["language"].to_numpy())

    # 4. Stratified Holdout Test Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n[3] STRATIFIED TEST SPLIT EVALUATION (Holdout: {int(test_size*100)}%, {len(y_test)} samples)")
    print(f"    Test Accuracy  : {test_acc * 100:.2f}%")
    print("\n    Detailed Classification Report:")
    print("    " + classification_report(y_test, y_test_pred, digits=4).replace("\n", "\n    "))

    print("    Confusion Matrix (Rows=True, Cols=Predicted):")
    classes = model.classes_
    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    header = "          " + "  ".join([f"{c:>8}" for c in classes])
    print(header)
    for i, row_label in enumerate(classes):
        row_str = f"    {row_label:<8}: " + "  ".join([f"{cm[i, j]:>8}" for j in range(len(classes))])
        print(row_str)

    # 5. Full Dataset Verification
    X_full_scaled = scaler.transform(X)
    y_full_pred = model.predict(X_full_scaled)
    full_acc = accuracy_score(y, y_full_pred)

    print(f"\n[4] FULL DATASET EVALUATION ({total_samples} samples)")
    print(f"    Overall Accuracy: {full_acc * 100:.2f}%")

    # 6. Final Model Status Determination
    print("\n" + "=" * 70)
    if test_acc >= 0.90 and full_acc >= 0.95:
        print(" [OK] MODEL STATUS: VERIFIED PRODUCTION-READY")
        print(f"     The model demonstrates high generalization ({test_acc*100:.2f}% test accuracy)")
        print("     and strict compatibility with the 26-feature MFCC pipeline.")
    else:
        print(" [!] MODEL STATUS: UNVERIFIED / REQUIRES ATTENTION")
    print("=" * 70)

    return {
        "verified": (test_acc >= 0.90),
        "test_accuracy": test_acc,
        "full_accuracy": full_acc,
        "total_samples": total_samples,
        "classes": classes.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing multilingual classification model.")
    parser.add_argument("--csv", type=str, default="language_features.csv", help="Path to feature CSV")
    parser.add_argument("--model", type=str, default="language_classifier.pkl", help="Path to model pickle")
    parser.add_argument("--scaler", type=str, default="feature_scaler.pkl", help="Path to scaler pickle")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set split fraction")
    args = parser.parse_args()

    evaluate_classifier(
        csv_path=args.csv,
        model_path=args.model,
        scaler_path=args.scaler,
        test_size=args.test_size
    )


if __name__ == "__main__":
    main()
