import streamlit as st
import pandas as pd
import joblib
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="❤️",
    layout="centered"
)

# -------------------------------------------------
# ADVANCED CSS (BACKGROUND + ANIMATIONS)
# -------------------------------------------------
st.markdown("""
<style>

/* ---- BACKGROUND ---- */
.stApp {
    background: linear-gradient(135deg, #020617, #020617, #0f172a);
    background-attachment: fixed;
}

/* ---- GLASS CARD ---- */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    animation: fadeUp 1s ease-in-out;
    margin-bottom: 25px;
}

/* ---- TEXT ---- */
.title {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
    color: #f8fafc;
    animation: glow 2s infinite alternate;
}
.subtitle {
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 40px;
}

/* ---- BUTTON ---- */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 16px;
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white;
    border: none;
    transition: all 0.4s ease;
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(239,68,68,0.8);
}

/* ---- ANIMATIONS ---- */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes glow {
    from { text-shadow: 0 0 10px #ef4444; }
    to { text-shadow: 0 0 25px #dc2626; }
}

label {
    color: #e5e7eb !important;
    font-weight: 600;
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
st.markdown('<div class="title">❤️ Heart Disease AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Next-gen AI system for early heart risk detection</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# INPUT CARD
# -------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("🧠 Health Information")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🚀 Analyze Heart Risk"):

    with st.spinner("Running AI model..."):
        time.sleep(1.5)

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
    # RESULT CARD (ANIMATED)
    # -------------------------------------------------
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📊 AI Prediction Result")

    st.metric("Risk Probability", f"{probability:.2f}%")
    st.progress(int(probability))

    if prediction == 1:
        st.error("""⚠️ HIGH RISK DETECTED  
Immediate medical consultation recommended.""")
    else:
        st.success("""✅ LOW RISK DETECTED  
Maintain a healthy lifestyle.""")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:#94a3b8;'>"
    "Built with ❤️ by Tejas | AI + Healthcare</p>",
    unsafe_allow_html=True
)
