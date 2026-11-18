import joblib
import numpy as np
import pandas as pd
import streamlit as st

THRESHOLD = 0.4  # chosen with cross-validation on the training data

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")


@st.cache_resource
def load_model():
    # Runs once, then Streamlit keeps the loaded model in memory
    return joblib.load("diabetes_model.joblib")


def optional_number(label, min_value, max_value, default, step, key):
    """A number box with a 'not measured' checkbox.

    In the training data, a 0 in these columns meant 'not measured'.
    Here we send NaN instead, and the model's imputer fills in the
    median it learned from the training data.
    """
    unknown = st.checkbox(f"{label}: not measured", key=f"{key}_unknown")
    value = st.number_input(label, min_value=min_value, max_value=max_value,
                            value=default, step=step, key=key, disabled=unknown)
    return np.nan if unknown else value


model = load_model()

st.title("Diabetes Predictor")
st.write("Enter a patient's measurements to get the model's estimated "
         "probability of diabetes.")
st.warning("This is a learning project, not a medical tool.")

left, right = st.columns(2)

with left:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20,
                                  value=2, step=1)
    age = st.number_input("Age (years)", min_value=21, max_value=90,
                          value=40, step=1)
    dpf = st.number_input("Diabetes pedigree function", min_value=0.05,
                          max_value=2.5, value=0.5, step=0.01)
    bmi = optional_number("BMI", 15.0, 70.0, 30.0, 0.1, "bmi")

with right:
    glucose = optional_number("Glucose (mg/dL)", 40, 250, 120, 1, "glucose")
    blood_pressure = optional_number("Blood pressure (mm Hg)", 30, 130, 70, 1, "bp")
    skin = optional_number("Skin thickness (mm)", 5, 100, 20, 1, "skin")
    insulin = optional_number("Insulin (mu U/ml)", 10, 900, 80, 1, "insulin")

if st.button("Predict", type="primary"):
    # Column order and names must match what the model was trained on
    row = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin,
                         insulin, bmi, dpf, age]], columns=FEATURES)

    prob = model.predict_proba(row)[0, 1]

    st.metric("Estimated probability of diabetes", f"{prob:.0%}")
    st.progress(float(prob))

    if prob >= THRESHOLD:
        st.error(f"Flagged as likely diabetic (probability is at or above "
                 f"the {THRESHOLD} threshold).")
    else:
        st.success(f"Not flagged (probability is below the {THRESHOLD} threshold).")

    n_missing = int(row.isna().sum().sum())
    if n_missing:
        st.info(f"{n_missing} value(s) were marked 'not measured', so the model "
                "filled them with typical values from its training data. "
                "The estimate is less reliable.")

    st.caption("On the held-out test set this model caught about 65% of "
               "diabetic patients, with about 61% of its flags correct.")

with st.expander("What does the model rely on?"):
    weights = pd.Series(model.named_steps["clf"].coef_[0], index=FEATURES)
    st.bar_chart(weights.sort_values())
    st.write("Bars to the right push the prediction toward diabetic, bars to "
             "the left push away. Glucose has the strongest effect. These "
             "describe the model, not what causes diabetes.")
