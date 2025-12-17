import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.main {
    background-color: #f5f7fb;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.title {
    font-size: 38px;
    font-weight: 700;
    color: #0f172a;
    text-align: center;
}
.subtitle {
    font-size: 16px;
    color: #475569;
    text-align: center;
    margin-bottom: 30px;
}
label {
    font-weight: 600 !important;
}
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #dc2626, #b91c1c);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ---------------- HEADER ----------------
st.markdown('<div class="title">❤️ Heart Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered health risk analysis by Tejas</div>', unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Heart Disease Risk"):

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

    st.markdown("<br>", unsafe_allow_html=True)

  if prediction == 1:
    st.error("""⚠️ **High Risk of Heart Disease**
Please consult a medical professional.""")
else:
    st.success("""✅ **Low Risk of Heart Disease**
Keep maintaining a healthy lifestyle!""")

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center; color:gray; margin-top:40px;'>"
    "Made with ❤️ using Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
