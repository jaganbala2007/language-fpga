"""
Robust Classifier Training and Validation Pipeline
FPGA-Based Multilingual AI Receptionist

Trains an MLP classifier (26 -> 16 -> 3) on the 26-feature dataset
with stratified train/test split, comprehensive evaluation metrics,
and safe model persistence.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def train(
    csv_path: str = "language_features.csv",
    output_model: str = "language_classifier.pkl",
    output_scaler: str = "feature_scaler.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    hidden_layer: int = 16,
    max_iter: int = 300,
    overwrite: bool = False
):
    print("=" * 60)
    print("  TRAINING FPGA-COMPATIBLE MULTILINGUAL MLP CLASSIFIER")
    print("=" * 60)

    if not os.path.exists(csv_path):
        alt = os.path.join("FPGA_Multilingual_Receptionist", csv_path)
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    if not overwrite and os.path.exists(output_model):
        print(f"[!] Warning: Model file '{output_model}' already exists.")
        print("    Use --overwrite flag if you explicitly wish to replace it.")
        response = input("Overwrite existing model? (y/N): ").strip().lower()
        if response != 'y':
            print("Training cancelled without modifying existing artifacts.")
            return

    print(f"[*] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "language"]
    
    X = np.asarray(df[feature_cols].to_numpy(), dtype=np.float64)
    y = np.asarray(df["language"].to_numpy())

    print(f"[*] Total dataset samples : {len(df)}")
    print(f"[*] Features per sample   : {len(feature_cols)}")
    print(f"[*] Class distribution    :\n{df['language'].value_counts().to_string()}")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n[*] Training set samples  : {len(X_train)}")
    print(f"[*] Testing set samples   : {len(X_test)}")

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP Architecture (26 -> 16 -> 3)
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer,),
        activation="relu",
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1
    )

    print(f"\n[*] Training MLP Classifier ({len(feature_cols)} -> {hidden_layer} -> 3)...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n[OK] Validation Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    # Save only after validation check
    if accuracy >= 0.90:
        joblib.dump(model, output_model)
        joblib.dump(scaler, output_scaler)
        print(f"\n[OK] Model successfully saved to : {output_model}")
        print(f"[OK] Scaler successfully saved to: {output_scaler}")
    else:
        print(f"\n[!] Model accuracy ({accuracy*100:.2f}%) did not meet 90% threshold. Not saved.")


def main():
    parser = argparse.ArgumentParser(description="Train multilingual audio MLP classifier.")
    parser.add_argument("--csv", type=str, default="language_features.csv", help="Input features CSV")
    parser.add_argument("--model-out", type=str, default="language_classifier.pkl", help="Output model path")
    parser.add_argument("--scaler-out", type=str, default="feature_scaler.pkl", help="Output scaler path")
    parser.add_argument("--hidden", type=int, default=16, help="Number of neurons in hidden layer")
    parser.add_argument("--max-iter", type=int, default=300, help="Max training iterations")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing model files")
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        output_model=args.model_out,
        output_scaler=args.scaler_out,
        hidden_layer=args.hidden,
        max_iter=args.max_iter,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
