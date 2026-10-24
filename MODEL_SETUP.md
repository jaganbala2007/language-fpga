# Model Setup, Validation, and Training Guide

## 1. Machine Learning Model Architecture

- **Classifier:** `sklearn.neural_network.MLPClassifier`
- **Feature Vector:** 26 continuous features
  - Features 1–13: MFCC Means ($\mu_{\text{mfcc}}$)
  - Features 14–26: MFCC Standard Deviations ($\sigma_{\text{mfcc}}$)
- **Input Dimension:** 26
- **Hidden Layer:** 16 neurons with ReLU activation
- **Output Layer:** 3 classes (`English`, `Hindi`, `Tamil`)
- **Total Trainable Parameters:** 483 (416 $W_1$ + 16 $b_1$ + 48 $W_2$ + 3 $b_2$)

---

## 2. Dataset Properties (`language_features.csv`)

- **Total Samples:** 7,089
- **English:** 2,602 samples (36.7%)
- **Tamil:** 2,367 samples (33.4%)
- **Hindi:** 2,120 samples (29.9%)
- **Data Quality:** 0 missing values, 0 duplicate rows

---

## 3. Model Evaluation Command

To evaluate the existing trained model and scaler without overwriting:
```bash
python src/evaluate_model.py
```

### Verified Evaluation Results:
- **Full Dataset Accuracy:** `99.28%`
- **Stratified Test Split Accuracy (20% holdout):** `98.17%`
- **Confusion Matrix (Test Set, 1,418 samples):**
  - English: 517 / 521 correct (Precision: 99.23%, Recall: 99.23%)
  - Hindi: 408 / 424 correct (Precision: 98.55%, Recall: 96.23%)
  - Tamil: 467 / 473 correct (Precision: 96.69%, Recall: 98.73%)

---

## 4. Retraining the Model

To retrain the classifier with custom hyperparameters or new data:
```bash
python src/train_classifier.py --csv language_features.csv --hidden 16 --max-iter 300 --overwrite
```
The script will validate the model on a 20% stratified holdout split and only persist artifacts if accuracy $\ge 90.0\%$.
