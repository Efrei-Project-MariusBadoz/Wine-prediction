import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Classification de la Qualité du Vin (8 variables)",
    page_icon="🍷",
    layout="wide"
)

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("best_knn_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("🍷 Évaluation de la Qualité du Vin (Pipeline 8 Variables)")
st.markdown("Classification supervisée par **k-NN optimisé** ($k=7$, pondération par l'inverse de la distance). Les **8 variables physico-chimiques** ont été sélectionnées selon le seuil de **90% de variance cumulée (ACP)** et l'**Information Mutuelle**.")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Variables motrices principales")
    alcohol = st.slider("Degré d'alcool (% vol)", min_value=8.0, max_value=15.0, value=11.5, step=0.1)
    density = st.slider("Densité (g/cm³)", min_value=0.985, max_value=1.040, value=0.993, step=0.001, format="%.3f")
    volatile_acidity = st.slider("Acidité volatile (g(acide acétique)/dm³)", min_value=0.08, max_value=1.60, value=0.26, step=0.01)
    chlorides = st.slider("Chlorures (g(NaCl)/dm³)", min_value=0.01, max_value=0.60, value=0.045, step=0.005, format="%.3f")

with col_right:
    st.subheader("Composantes non-linéaires et équilibre")
    total_sulfur_dioxide = st.slider("Dioxyde de soufre total (mg/dm³)", min_value=6.0, max_value=350.0, value=115.0, step=1.0)
    free_sulfur_dioxide = st.slider("Dioxyde de soufre libre (mg/dm³)", min_value=1.0, max_value=150.0, value=30.0, step=1.0)
    citric_acid = st.slider("Acide citrique (g/dm³)", min_value=0.0, max_value=1.7, value=0.34, step=0.01)
    residual_sugar = st.slider("Sucre résiduel (g/dm³)", min_value=0.6, max_value=30.0, value=3.2, step=0.1)

st.markdown("---")

if st.button("🔍 Évaluer la qualité du vin", type="primary", use_container_width=True):
    # Ordre strict des 8 features attendues par le StandardScaler
    features = pd.DataFrame([{
        "alcohol": alcohol,
        "density": density,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "chlorides": chlorides,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "volatile_acidity": volatile_acidity,
        "free_sulfur_dioxide": free_sulfur_dioxide
    }])

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    prob_good = probabilities[1]

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.success("### Résultat : Grand Vin (Qualité ≥ 7)")
            st.write(f"Confiance du modèle : **{prob_good * 100:.1f}%**")
        else:
            st.info("### Résultat : Vin Ordinaire (Qualité < 7)")
            st.write(f"Probabilité de grand vin : **{prob_good * 100:.1f}%**")

        st.progress(float(prob_good))

    with res_col2:
        st.write("**Profil de l'échantillon testé :**")
        st.write(f"- Alcool : **{alcohol}% vol** (médiane bons vins : ~11.5%)")
        st.write(f"- Densité : **{density} g/cm³** (médiane bons vins : ~0.992 g/cm³)")
        st.write(f"- Acidité volatile : **{volatile_acidity} g/L** (médiane bons vins : ~0.26 g/L)")
        st.write(f"- SO₂ total : **{total_sulfur_dioxide} mg/L**")
