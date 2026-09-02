import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Smart Manufacturing AI",
    page_icon="⚙️",
    layout="wide"
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "best_model.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "model_metadata.json"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


@st.cache_data
def load_metadata():

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return metadata


model = load_model()
metadata = load_metadata()


# ---------------------------------------------------------
# BUSINESS RULES
# ---------------------------------------------------------

def calculate_risk(probability):

    if probability >= 0.70:
        return "CRITICAL"

    elif probability >= 0.40:
        return "HIGH"

    elif probability >= 0.15:
        return "MEDIUM"

    else:
        return "LOW"


def calculate_health_score(probability):

    score = (1 - probability) * 100

    return round(score, 1)


def maintenance_recommendation(risk):

    recommendations = {

        "CRITICAL":
            "Stop machine and perform immediate maintenance inspection.",

        "HIGH":
            "Schedule maintenance as soon as possible and inspect critical components.",

        "MEDIUM":
            "Monitor machine closely and schedule preventive maintenance.",

        "LOW":
            "Machine appears healthy. Continue normal monitoring."
    }

    return recommendations[risk]


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("⚙️ Smart Manufacturing AI")

st.subheader(
    "Equipment Failure Prediction & Maintenance Decision Support"
)

st.markdown(
    """
    This application uses a **supervised machine learning model**
    trained on the AI4I 2020 Predictive Maintenance Dataset.

    Enter machine operating conditions below to predict the probability
    of equipment failure and receive a maintenance recommendation.
    """
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Machine Information")

machine_id = st.sidebar.text_input(
    "Machine ID",
    value="M001"
)

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)


# ---------------------------------------------------------
# INPUT PARAMETERS
# ---------------------------------------------------------

st.header("🔧 Machine Operating Conditions")

col1, col2 = st.columns(2)

with col1:

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=280.0,
        max_value=320.0,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=280.0,
        max_value=330.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=500,
        max_value=3000,
        value=1500,
        step=10
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )


with col2:

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )

    power = rotational_speed * torque

    st.metric(
        "Calculated Power [W]",
        f"{power:,.2f}"
    )

    temperature_difference = (
        process_temperature - air_temperature
    )

    st.metric(
        "Temperature Difference [K]",
        f"{temperature_difference:.2f}"
    )


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

st.divider()

predict_button = st.button(
    "🚀 Predict Machine Failure",
    use_container_width=True
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if predict_button:

    # Create dataframe using EXACT training feature names

    input_data = pd.DataFrame({

        "Air temperature [K]": [
            air_temperature
        ],

        "Process temperature [K]": [
            process_temperature
        ],

        "Rotational speed [rpm]": [
            rotational_speed
        ],

        "Torque [Nm]": [
            torque
        ],

        "Tool wear [min]": [
            tool_wear
        ],

        "Temperature difference [K]": [
            temperature_difference
        ],

        "Power [W]": [
            power
        ],

        "Type": [
            machine_type
        ]
    })


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = model.predict(
        input_data
    )[0]


    probability_percent = probability * 100

    risk = calculate_risk(probability)

    health_score = calculate_health_score(
        probability
    )

    recommendation = maintenance_recommendation(
        risk
    )


    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    st.header("📊 Prediction Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Failure Probability",
            f"{probability_percent:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Level",
            risk
        )

    with col3:

        st.metric(
            "Machine Health",
            f"{health_score}/100"
        )

    with col4:

        status = "FAILURE RISK" if prediction == 1 else "NORMAL"

        st.metric(
            "Prediction",
            status
        )


    # -----------------------------------------------------
    # RISK MESSAGE
    # -----------------------------------------------------

    if risk == "CRITICAL":

        st.error(
            f"🚨 CRITICAL RISK: {recommendation}"
        )

    elif risk == "HIGH":

        st.warning(
            f"⚠️ HIGH RISK: {recommendation}"
        )

    elif risk == "MEDIUM":

        st.warning(
            f"🟡 MEDIUM RISK: {recommendation}"
        )

    else:

        st.success(
            f"🟢 LOW RISK: {recommendation}"
        )


    # -----------------------------------------------------
    # GAUGE
    # -----------------------------------------------------

    st.subheader("Machine Health Score")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={
                "text": "Health Score"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },

                "steps": [

                    {
                        "range": [0, 40],
                        "color": "red"
                    },

                    {
                        "range": [40, 70],
                        "color": "orange"
                    },

                    {
                        "range": [70, 100],
                        "color": "green"
                    }
                ]
            }
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------------------
    # INPUT SUMMARY
    # -----------------------------------------------------

    st.subheader("Machine Input Summary")

    display_data = input_data.T

    display_data.columns = ["Value"]

    st.dataframe(
        display_data,
        use_container_width=True
    )


    # -----------------------------------------------------
    # MAINTENANCE DECISION
    # -----------------------------------------------------

    st.subheader("🛠️ Maintenance Decision")

    st.info(
        f"""
        **Machine:** {machine_id}

        **Risk Level:** {risk}

        **Failure Probability:** {probability_percent:.2f}%

        **Health Score:** {health_score}/100

        **Recommendation:** {recommendation}
        """
    )


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

with st.expander("🤖 Model Information"):

    st.write(
        f"**Best Model:** {metadata['best_model_name']}"
    )

    metrics = metadata["test_metrics"]

    st.write(
        f"**Accuracy:** {metrics['accuracy']:.4f}"
    )

    st.write(
        f"**Precision:** {metrics['precision']:.4f}"
    )

    st.write(
        f"**Recall:** {metrics['recall']:.4f}"
    )

    st.write(
        f"**F1 Score:** {metrics['f1']:.4f}"
    )

    st.write(
        f"**ROC-AUC:** {metrics['roc_auc']:.4f}"
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Smart Manufacturing AI | Supervised Machine Learning | XGBoost"
)