import os
import joblib
import numpy as np
import pytest

MODEL_PATH = "best_knn_model.joblib"
SCALER_PATH = "scaler.joblib"

def test_artifacts_exist():
    assert os.path.exists(MODEL_PATH), "Le fichier du modele best_knn_model.joblib est introuvable."
    assert os.path.exists(SCALER_PATH), "Le fichier du scaler scaler.joblib est introuvable."

def test_load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    assert hasattr(model, "predict"), "L'objet charge n'est pas un modele scikit-learn valide."
    assert hasattr(scaler, "transform"), "L'objet charge n'est pas un StandardScaler scikit-learn valide."
    assert scaler.n_features_in_ == 8, f"Le scaler attend 8 features, {scaler.n_features_in_} detectees."

def test_prediction_output_shape_and_type():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # 8 features: alcohol, density, total_sulfur_dioxide, chlorides, citric_acid, residual_sugar, volatile_acidity, free_sulfur_dioxide
    dummy_input = np.array([[11.5, 0.993, 115.0, 0.045, 0.34, 3.2, 0.26, 30.0]])
    scaled_input = scaler.transform(dummy_input)
    
    pred = model.predict(scaled_input)
    prob = model.predict_proba(scaled_input)
    
    assert pred.shape == (1,)
    assert pred[0] in [0, 1]
    assert prob.shape == (1, 2)
    assert np.isclose(prob.sum(), 1.0, atol=1e-5)

def test_dimension_mismatch_raises_error():
    scaler = joblib.load(SCALER_PATH)
    invalid_input = np.array([[11.5, 0.993, 115.0, 0.045]]) # 4 variables au lieu de 8
    with pytest.raises(ValueError):
        scaler.transform(invalid_input)
