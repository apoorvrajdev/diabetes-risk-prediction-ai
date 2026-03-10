import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="AI Diabetes Risk Analyzer",
    page_icon="🩺",
    layout="wide",
)


MODEL_PATH = Path("diabetes_model.pkl")


@st.cache_resource
def load_model():
    """Load the trained model from the current project folder."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_feature_array(
    age: int,
    hypertension: int,
    heart_disease: int,
    bmi: float,
    hba1c: float,
    glucose: int,
    gender: str,
    smoking: str,
) -> np.ndarray:
    """Encode UI inputs into the 13-feature array expected by the model."""
    gender_male = 1 if gender == "Male" else 0
    gender_other = 1 if gender == "Other" else 0

    smoke_current = 1 if smoking == "current" else 0
    smoke_ever = 1 if smoking == "ever" else 0
    smoke_former = 1 if smoking == "former" else 0
    smoke_never = 1 if smoking == "never" else 0
    smoke_not_current = 1 if smoking == "not current" else 0

    features = np.array(
        [
            [
                age,
                hypertension,
                heart_disease,
                bmi,
                hba1c,
                glucose,
                gender_male,
                gender_other,
                smoke_current,
                smoke_ever,
                smoke_former,
                smoke_never,
                smoke_not_current,
            ]
        ],
        dtype=float,
    )
    return features


def render_prediction_page(model) -> None:
    st.title("🩺 AI Diabetes Risk Analyzer")
    st.write("Enter patient details to estimate diabetes risk.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=1, max_value=100, value=30)
        hypertension = st.selectbox("Hypertension", options=[0, 1])
        heart_disease = st.selectbox("Heart Disease", options=[0, 1])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    with col2:
        hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
        glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120, step=1)
        gender = st.selectbox("Gender", options=["Female", "Male", "Other"])
        smoking = st.selectbox(
            "Smoking History",
            options=["never", "current", "former", "ever", "not current", "No Info"],
        )

    features = build_feature_array(
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        bmi=bmi,
        hba1c=hba1c,
        glucose=glucose,
        gender=gender,
        smoking=smoking,
    )

    st.subheader("📋 Patient Summary")
    summary_df = pd.DataFrame(
        {
            "Feature": [
                "Age",
                "Hypertension",
                "Heart Disease",
                "BMI",
                "HbA1c",
                "Blood Glucose",
                "Gender",
                "Smoking",
            ],
            "Value": [
                age,
                hypertension,
                heart_disease,
                bmi,
                hba1c,
                glucose,
                gender,
                smoking,
            ],
        }
    )
    st.table(summary_df)

    if st.button("Predict Diabetes Risk", type="primary"):
        prediction = model.predict(features)
        probability = float(model.predict_proba(features)[0][1] * 100)

        st.subheader("🧠 AI Prediction Result")

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                title={"text": "Diabetes Risk %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 30], "color": "green"},
                        {"range": [30, 60], "color": "yellow"},
                        {"range": [60, 100], "color": "red"},
                    ],
                },
            )
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.subheader("📊 Prediction Confidence")
        confidence_df = pd.DataFrame(
            {
                "Class": ["Healthy", "Diabetes Risk"],
                "Probability": [100 - probability, probability],
            }
        )
        st.bar_chart(confidence_df.set_index("Class"))

        if int(prediction[0]) == 1:
            st.error(f"⚠ High Risk of Diabetes ({probability:.2f}%)")
        else:
            st.success(f"✅ Low Risk of Diabetes ({probability:.2f}%)")

        st.subheader("🩺 Medical Interpretation")
        if probability < 30:
            st.success("Low diabetes risk. Maintain a healthy lifestyle.")
        elif probability < 60:
            st.warning("Moderate diabetes risk. Consider improving diet and exercise.")
        else:
            st.error("High diabetes risk detected. Medical consultation recommended.")

        st.subheader("💡 Lifestyle Recommendations")
        st.write(
            """
- Maintain a healthy BMI (18.5 to 24.9)
- Exercise at least 30 minutes daily
- Reduce sugar and refined carbohydrate intake
- Monitor blood glucose regularly
- Avoid smoking and excessive alcohol
"""
        )

        st.warning(
            "⚠ This AI tool estimates diabetes risk based on machine learning patterns. "
            "It is NOT a medical diagnosis. Always consult a healthcare professional."
        )


def render_model_information_page() -> None:
    st.title("📊 Model Information")
    st.write(
        """
**Model Type:** RandomForest / ML classifier

**Features Used:**
- Age
- Hypertension
- Heart Disease
- BMI
- HbA1c Level
- Blood Glucose Level
- Gender
- Smoking History

**Dataset Size:** ~175,000 samples after SMOTE balancing

**Evaluation Metric:** ROC-AUC
"""
    )
    st.info(
        "This model was trained using supervised machine learning to estimate diabetes risk "
        "based on patient clinical features."
    )


def main() -> None:
    st.sidebar.title("🧠 AI Health Dashboard")
    page = st.sidebar.radio("Navigation", ["Predict Diabetes", "Model Information"])

    if not MODEL_PATH.exists():
        st.error("Model file 'diabetes_model.pkl' was not found in the current folder.")
        st.stop()

    model = load_model()

    if page == "Predict Diabetes":
        render_prediction_page(model)
    elif page == "Model Information":
        render_model_information_page()


if __name__ == "__main__":
    main()
