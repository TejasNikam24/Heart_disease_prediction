import streamlit as st
import pandas as pd
import joblib
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="HeartCare AI",
    page_icon="🫀",
    layout="centered"
)

# -------------------------------------------------
# PREMIUM CSS (OFFICIAL HEALTH-TECH STYLE)
# -------------------------------------------------
st.markdown("""
<style>

/* ---- GLOBAL BACKGROUND ---- */
.stApp {
    background: linear-gradient(135deg, #020617, #020617, #0f172a);
    font-family: 'Inter', sans-serif;
}

/* ---- HERO ---- */
.hero-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    color: #f8fafc;
    margin-top: 10px;
}
.hero-subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 16px;
    margin-bottom: 45px;
}

/* ---- GLASS CARD ---- */
.card {
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(18px);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 28px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.4);
    animation: fadeSlide 0.9s ease;
}

/* ---- BUTTON ---- */
.stButton > button {
    width: 100%;
    height: 56px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white;
    border: none;
    transition: all 0.35s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(37,99,235,0.6);
}

/* ---- TEXT ---- */
label {
    color: #e5e7eb !important;
    font-weight: 600;
}
h3 {
    color: #f1f5f9;
}

/* ---- ANIMATION ---- */
@keyframes fadeSlide {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
st.markdown('<div class="hero-title">🫀 HeartCare AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Clinical-grade AI system for early heart disease risk assessment</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# INPUT CARD
# -------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Patient Health Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col2:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("Analyze Heart Risk"):

    with st.spinner("Processing clinical data..."):
        time.sleep(1.2)

        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] * 100

    # -------------------------------------------------
    # RESULT CARD
    # -------------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("AI Risk Assessment")

    st.metric("Estimated Heart Disease Risk", f"{probability:.2f}%")
    st.progress(int(probability))

    if prediction == 1:
        st.error("""High risk detected.
Please seek professional medical consultation.""")
    else:
        st.success("""Low risk detected.
Continue maintaining a healthy lifestyle.""")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:#94a3b8;'>"
    "HeartCare AI • Built with Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
