import streamlit as st
import pandas as pd
import joblib
import logging
import warnings
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning

logger = logging.getLogger(__name__)

COURSES = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening)",
}

FEATURES = [
    "Course", "Tuition fees up to date", "Scholarship holder",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "student_dropout_final_pipeline.pkl"

@st.cache_resource
def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        pipeline = joblib.load(MODEL_PATH)
    if list(pipeline.feature_names_in_) != FEATURES:
        raise ValueError("Model input schema does not match the application.")
    if list(pipeline.classes_) != [0, 1, 2]:
        raise ValueError("Model classes do not match the application.")
    return pipeline

try:
    model = load_model()
except Exception:
    logger.exception("Model initialization failed")
    st.error(
        "Unable to load the model. Make sure "
        "'student_dropout_final_pipeline.pkl' is in the same folder as app.py "
        "and install the exact dependencies in requirements.txt. See server logs for details."
    )
    st.stop()

# LabelEncoder used during training produced:
# 0 = Dropout, 1 = Enrolled, 2 = Graduate
CLASS_NAMES = {
    0: "Dropout",
    1: "Enrolled",
    2: "Graduate"
}

st.title("🎓 Student Dropout Prediction")
st.markdown(
    "### Predict a student's academic outcome using the trained Machine Learning model."
)
st.info("Use completed first- and second-semester records. The model uses 10 features.")

st.subheader("Enter Student Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        course = st.selectbox(
            "Course",
            options=list(COURSES),
            index=1,
            format_func=lambda code: f"{code} - {COURSES[code]}",
            help="Numeric course code from the original dataset."
        )

        tuition_fees = st.selectbox(
            "Tuition fees up to date",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        scholarship = st.selectbox(
            "Scholarship holder",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        first_sem_enrolled = st.number_input(
            "1st Semester - Units Enrolled",
            min_value=0, max_value=30, value=6, step=1
        )

        first_sem_approved = st.number_input(
            "1st Semester - Units Approved",
            min_value=0, max_value=30, value=5, step=1
        )

    with col2:
        second_sem_credited = st.number_input(
            "2nd Semester - Units Credited",
            min_value=0, max_value=30, value=0, step=1
        )

        second_sem_enrolled = st.number_input(
            "2nd Semester - Units Enrolled",
            min_value=0, max_value=30, value=6, step=1
        )

        second_sem_evaluations = st.number_input(
            "2nd Semester - Number of Evaluations",
            min_value=0, value=6, step=1,
            help="Count all evaluations, including repeated attempts. This may exceed units enrolled."
        )

        second_sem_approved = st.number_input(
            "2nd Semester - Units Approved",
            min_value=0, max_value=30, value=5, step=1
        )

        second_sem_grade = st.number_input(
            "2nd Semester - Grade",
            min_value=0.0, max_value=20.0, value=12.5, step=0.1
        )

    submitted = st.form_submit_button(
        "🔍 Predict Student Outcome",
        width="stretch"
    )

if submitted:
    errors = []
    if first_sem_approved > first_sem_enrolled:
        errors.append("1st semester: approved units cannot exceed enrolled units.")
    if second_sem_approved > second_sem_enrolled:
        errors.append("2nd semester: approved units cannot exceed enrolled units.")
    if second_sem_credited > second_sem_enrolled:
        errors.append("2nd semester: credited units cannot exceed enrolled units.")
    if errors:
        for message in errors:
            st.error(message)
        st.stop()
    new_student = pd.DataFrame({
        "Course": [course],
        "Tuition fees up to date": [tuition_fees],
        "Scholarship holder": [scholarship],
        "Curricular units 1st sem (enrolled)": [first_sem_enrolled],
        "Curricular units 1st sem (approved)": [first_sem_approved],
        "Curricular units 2nd sem (credited)": [second_sem_credited],
        "Curricular units 2nd sem (enrolled)": [second_sem_enrolled],
        "Curricular units 2nd sem (evaluations)": [second_sem_evaluations],
        "Curricular units 2nd sem (approved)": [second_sem_approved],
        "Curricular units 2nd sem (grade)": [second_sem_grade]
    }, columns=FEATURES)

    try:
        prediction = model.predict(new_student)[0]
        predicted_outcome = CLASS_NAMES[int(prediction)]

        st.divider()
        st.subheader("Prediction Result")

        if predicted_outcome == "Graduate":
            st.success("🎓 Predicted Outcome: GRADUATE")
        elif predicted_outcome == "Enrolled":
            st.warning("📚 Predicted Outcome: ENROLLED")
        else:
            st.error("⚠️ Predicted Outcome: DROPOUT")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(new_student)[0]

            probability_df = pd.DataFrame({
                "Outcome": [
                    CLASS_NAMES[int(class_id)]
                    for class_id in model.classes_
                ],
                "Probability (%)": (probabilities * 100).round(2)
            })

            st.subheader("Prediction Probabilities")
            st.dataframe(
                probability_df,
                width="stretch",
                hide_index=True
            )

    except Exception:
        logger.exception("Prediction failed")
        st.error("Prediction could not be completed. Please contact the app administrator.")

st.divider()
st.caption("Student Dropout Prediction • Machine Learning Project")
