import streamlit as st
from src.utility import load_model, prepare_input

# Load model, scaler, and encoders
bundle = load_model("model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
encoders = bundle["encoders"]

# Streamlit UI
st.title("Job Suitability Predictor")

# Input widgets
age = st.slider("Age", 18, 60, 25)
gender = st.selectbox("Gender", encoders["Gender"].classes_)
education = st.selectbox("Education", encoders["Education"].classes_)
skill = st.selectbox("Skill Level", encoders["Skill_Level"].classes_)
experience = st.slider("Experience (Years)", 0, 20, 2)

# Transform categorical inputs using encoders
gender_val = encoders["Gender"].transform([gender])[0]
education_val = encoders["Education"].transform([education])[0]
skill_val = encoders["Skill_Level"].transform([skill])[0]

# Prepare and scale input
input_data = prepare_input([age, gender_val, education_val, skill_val, experience])
input_data = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    if pred == 1:
        st.success("Suitable ✅")  # green
    else:
        st.error("Not Suitable ❌")  # red
