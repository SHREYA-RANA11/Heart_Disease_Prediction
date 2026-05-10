import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

h1, h2, h3 {
    color: white;
}

.card {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.1);
    margin-bottom: 20px;
}

.result-success {
    background-color: #14532d;
    padding: 20px;
    border-radius: 15px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    color: white;
}

.result-danger {
    background-color: #7f1d1d;
    padding: 20px;
    border-radius: 15px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    color: white;
}

.small-text {
    color: #cbd5e1;
    font-size: 15px;
}

.stButton > button {
    width: 100%;
    background-color: #ef4444;
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #dc2626;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL FILES ----------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>❤️ Heart Disease Prediction System</h1>
<p style='text-align:center;' class='small-text'>
Predict heart disease risk using Machine Learning
</p>
""", unsafe_allow_html=True)

st.write("")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# ---------------- LEFT SIDE ----------------
with col1:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("👤 Personal Information")

    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT SIDE ----------------
with col2:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🩺 Medical Information")

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm/Hg)",
        80,
        200,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        100,
        600,
        200
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICT BUTTON ----------------
predict_btn = st.button("🔍 Predict Heart Disease Risk")

# ---------------- PREDICTION ----------------
if predict_btn:

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

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns properly
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    st.write("")

    # ---------------- RESULT ----------------
    if prediction == 1:

        st.markdown("""
        <div class='result-danger'>
        ⚠️ High Risk of Heart Disease
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class='result-success'>
        ✅ Low Risk of Heart Disease
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.write("")
st.write("")

st.markdown("""
<hr>

<p style='text-align:center;' class='small-text'>
Developed with ❤️ by Shreya Rana
</p>
""", unsafe_allow_html=True)