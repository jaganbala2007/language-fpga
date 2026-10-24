"""
Unit Tests: Model and Scaler Artifact Validation
"""

import os
import sys
import pytest
import numpy as np
import joblib

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import predict_language_from_features, LANGUAGE_CLASSES


def test_model_and_scaler_loading_and_prediction():
    model_path = "language_classifier.pkl" if os.path.exists("language_classifier.pkl") else "FPGA_Multilingual_Receptionist/language_classifier.pkl"
    scaler_path = "feature_scaler.pkl" if os.path.exists("feature_scaler.pkl") else "FPGA_Multilingual_Receptionist/feature_scaler.pkl"

    assert os.path.exists(model_path)
    assert os.path.exists(scaler_path)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    assert model.n_features_in_ == 26
    assert scaler.n_features_in_ == 26
    assert set(model.classes_) == set(LANGUAGE_CLASSES)

    # Test random 26-element feature vector
    dummy_feat = np.ones(26, dtype=np.float64) * 5.0
    pred, conf, scores = predict_language_from_features(dummy_feat, model, scaler)

    assert pred in LANGUAGE_CLASSES
    assert 0.0 <= conf <= 1.0
    assert len(scores) == 3
    assert abs(sum(scores.values()) - 1.0) < 1e-4
