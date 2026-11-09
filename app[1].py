
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

@st.cache_data
def load_metrics():
    with open("model_metrics.json", "r") as f:
        return json.load(f)

model = load_model()
data = load_data()
metrics = load_metrics()

st.title("🎓 Student Performance Prediction System")
st.caption("Academic Risk Identification and Decision Support Dashboard")
st.divider()

# STUDENT STATISTICS
st.header("Student Statistics")

total = len(data)
average = data["G3"].mean()
at_risk = int((data["G3"] < 10).sum())
not_at_risk = int((data["G3"] >= 10).sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Students", total)
c2.metric("Average Final Grade", round(average, 2))
c3.metric("Historical At Risk", at_risk)
c4.metric("Historical Not At Risk", not_at_risk)

# MODEL EVALUATION
st.divider()
st.header("Model Evaluation Results")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
m2.metric("Precision", f"{metrics['precision']*100:.2f}%")
m3.metric("Recall", f"{metrics['recall']*100:.2f}%")
m4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")

st.caption("Metrics are calculated from the held-out test dataset.")

# DATASET PREVIEW
st.divider()
st.header("Student Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

# CHARTS
st.divider()
st.header("Student Performance Analysis")

left, right = st.columns(2)

with left:
    st.subheader("Student Risk Distribution")
    st.bar_chart(pd.Series({
        "At Risk": at_risk,
        "Not At Risk": not_at_risk
    }))

with right:
    st.subheader("Average Grades")
    st.bar_chart(pd.Series({
        "G1": data["G1"].mean(),
        "G2": data["G2"].mean(),
        "G3": data["G3"].mean()
    }))

# STUDENT PREDICTION
st.divider()
st.header("Predict Student Academic Risk")

st.write(
    "Upload a CSV file with the same predictor columns used "
    "during model training. Do not include G3."
)

template = data.drop(columns=["G3"]).head(5)

st.download_button(
    "Download Sample Input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv"
)

uploaded = st.file_uploader(
    "Upload Student CSV File",
    type=["csv"]
)

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        required = list(model.feature_names_in_)

        missing = [
            col for col in required
            if col not in new_data.columns
        ]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))

        elif new_data.empty:
            st.warning("Uploaded CSV has no records.")

        else:
            prediction = model.predict(new_data[required])

            results = new_data.copy()

            results["Prediction"] = [
                "AT RISK" if p == 1 else "NOT AT RISK"
                for p in prediction
            ]

            st.subheader("Prediction Results")
            st.dataframe(results, use_container_width=True)

            risk_count = int((prediction == 1).sum())
            safe_count = int((prediction == 0).sum())

            a, b = st.columns(2)
            a.metric("Predicted AT RISK", risk_count)
            b.metric("Predicted NOT AT RISK", safe_count)

            st.subheader("Predicted Risk Distribution")
            st.bar_chart(pd.Series({
                "AT RISK": risk_count,
                "NOT AT RISK": safe_count
            }))

            st.download_button(
                "Download Prediction Results",
                results.to_csv(index=False),
                "student_predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Unable to process CSV: {e}")

# RESPONSIBLE USE
st.divider()

st.warning(
    "IMPORTANT: Predictions are estimates for classroom "
    "demonstration only. They are not diagnoses and should not "
    "be the sole basis for real student decisions. Faculty members "
    "should review student circumstances before taking action."
)

st.caption(
    "UCI Student Performance Mathematics Dataset | "
    "Random Forest Classification"
)
