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
# PREMIUM ENTERPRISE CSS
# -------------------------------------------------
st.markdown("""
<style>

/* ---- BACKGROUND IMAGE ---- */
.stApp {
    background:
        linear-gradient(rgba(2,6,23,0.85), rgba(2,6,23,0.85)),
        url("https://images.unsplash.com/photo-1580281657521-6f9c3c58a6b1");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
}

/* ---- HERO ---- */
.hero {
    text-align: center;
    margin-bottom: 45px;
    animation: fadeDown 1s ease;
}
.hero h1 {
    font-size: 46px;
    font-weight: 800;
    color: #f8fafc;
}
.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

/* ---- GLASS PANEL ---- */
.panel {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.45);
    animation: fadeUp 0.9s ease;
}

/* ---- PANEL TITLE ---- */
.panel-title {
    font-size: 20px;
    font-weight: 700;
    color: #e5e7eb;
    margin-bottom: 20px;
    border-left: 4px solid #2563eb;
    padding-left: 12px;
}

/* ---- LABELS ---- */
label {
    color: #e5e7eb !important;
    font-weight: 600;
}

/* ---- BUTTON ---- */
.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white;
    border: none;
    transition: all 0.35s ease;
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(37,99,235,0.6);
}

/* ---- ANIMATIONS ---- */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
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
st.markdown("""
<div class="hero">
    <h1>🫀 HeartCare AI</h1>
    <p>Advanced clinical intelligence for early heart disease risk assessment</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PATIENT DETAILS PANEL
# -------------------------------------------------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Patient Information</div>', unsafe_allow_html=True)

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
# PREDICTION PANEL
# -------------------------------------------------
if st.button("Analyze Heart Health"):

    with st.spinner("Analyzing clinical parameters..."):
        time.sleep(1.4)

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

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">AI Risk Assessment</div>', unsafe_allow_html=True)

    st.metric("Estimated Risk Level", f"{probability:.2f}%")
    st.progress(int(probability))

    if prediction == 1:
        st.error("High risk detected. Medical consultation is advised.")
    else:
        st.success("Low risk detected. Maintain a healthy lifestyle.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:#94a3b8;'>"
    "HeartCare AI • Clinical Decision Support System</p>",
    unsafe_allow_html=True
)
