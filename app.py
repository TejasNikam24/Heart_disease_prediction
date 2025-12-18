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
# MODERN CSS (NO EMPTY BARS)
# -------------------------------------------------
st.markdown("""
<style>

/* ---------- BACKGROUND ---------- */
.stApp {
    background:
        linear-gradient(rgba(2, 6, 23, 0.92), rgba(2, 6, 23, 0.92)),
        url("https://images.unsplash.com/photo-1580281657521-6f9c3c58a6b1");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ---------- MAIN HEADER ---------- */
.main-header {
    background: linear-gradient(90deg, #0f172a, #1e40af, #0ea5e9);
    border-radius: 22px;
    padding: 28px 22px;
    margin: 30px 0 35px 0;
    box-shadow: 0 25px 55px rgba(14,165,233,0.45);
    animation: fadeDown 0.9s ease;
}

.main-header h1 {
    text-align: center;
    font-size: 42px;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}

.main-header p {
    text-align: center;
    font-size: 15px;
    color: #bae6fd;
}

/* ---------- GLASS CARD ---------- */
.card {
    background: rgba(255,255,255,0.13);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 35px;
    box-shadow: 0 30px 65px rgba(0,0,0,0.45);
    animation: fadeUp 0.8s ease;
}

/* ---------- CARD TITLE ---------- */
.card-title {
    font-size: 21px;
    font-weight: 800;
    color: #e5f0ff;
    margin-bottom: 24px;
    border-left: 5px solid #38bdf8;
    padding-left: 14px;
}

/* ---------- INPUT LABEL ---------- */
label {
    color: #e5e7eb !important;
    font-weight: 600;
}

/* ---------- BUTTON ---------- */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 18px;
    font-size: 19px;
    font-weight: 700;
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: #ffffff;
    border: none;
    transition: all 0.35s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 18px 45px rgba(14,165,233,0.7);
}

/* ---------- ANIMATIONS ---------- */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(25px); }
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
# MAIN HEADING (ONLY ONE)
# -------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🫀 HeartCare AI</h1>
    <p>Advanced clinical intelligence for early heart disease risk assessment</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# INPUT CARD
# -------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Patient Medical Information</div>', unsafe_allow_html=True)

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
# PREDICTION (NO EXTRA EMPTY BAR)
# -------------------------------------------------
if st.button("Analyze Heart Health"):

    with st.spinner("AI analysis in progress..."):
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

    st.metric("Estimated Heart Disease Risk", f"{probability:.2f}%")
    st.progress(int(probability))

    if prediction == 1:
        st.error("⚠️ High risk detected. Please consult a medical professional.")
    else:
        st.success("✅ Low risk detected. Maintain a healthy lifestyle.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:#94a3b8; margin-top:30px;'>"
    "HeartCare AI • AI-powered clinical decision support system</p>",
    unsafe_allow_html=True
)
