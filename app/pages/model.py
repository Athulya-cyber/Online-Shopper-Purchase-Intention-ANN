from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "Model"

@st.cache_resource
def load_prediction_assets():
    """Load the trained ANN, fitted preprocessor, and decision threshold."""

    model_path = MODEL_DIR / "online_shopper_ann.keras"
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    threshold_path = MODEL_DIR / "threshold.joblib"

    missing_files = [
        str(path)
        for path in (model_path, preprocessor_path, threshold_path)
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required model files are missing: " + ", ".join(missing_files)
        )

    trained_model = load_model(model_path)
    fitted_preprocessor = joblib.load(preprocessor_path)
    decision_threshold = float(joblib.load(threshold_path))

    return trained_model, fitted_preprocessor, decision_threshold


st.title("Purchase Intention Prediction")

st.write(
    """
    Enter the details of an online shopping session below.
    The trained ANN model will predict whether the session
    is likely to result in a purchase.
    """
)

st.divider()

try:
    model, preprocessor, threshold = load_prediction_assets()

except Exception as error:
    st.error(
        f"Unable to load the prediction files. Check that the Model folder "
        f"contains the trained model, preprocessor, and threshold files. "
        f"Details: {error}"
    )
    st.stop()


st.header("Page Visit Information")

col1, col2 = st.columns(2)

with col1:
    administrative = st.number_input(
        "Administrative Pages",
        min_value=0,
        value=0,
        step=1
    )

    informational = st.number_input(
        "Informational Pages",
        min_value=0,
        value=0,
        step=1
    )

    product_related = st.number_input(
        "Product Related Pages",
        min_value=0,
        value=1,
        step=1
    )

with col2:
    administrative_duration = st.number_input(
        "Administrative Duration",
        min_value=0.0,
        value=0.0
    )

    informational_duration = st.number_input(
        "Informational Duration",
        min_value=0.0,
        value=0.0
    )

    product_related_duration = st.number_input(
        "Product Related Duration",
        min_value=0.0,
        value=0.0
    )


st.divider()

st.header("Session Behaviour")

col1, col2 = st.columns(2)

with col1:
    bounce_rates = st.number_input(
        "Bounce Rate",
        min_value=0.0,
        value=0.0,
        format="%.5f"
    )

    page_values = st.number_input(
        "Page Values",
        min_value=0.0,
        value=0.0
    )

with col2:
    exit_rates = st.number_input(
        "Exit Rate",
        min_value=0.0,
        value=0.0,
        format="%.5f"
    )

    special_day = st.number_input(
        "Special Day",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1
    )


st.divider()

st.header("Visitor Information")

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox(
        "Month",
        ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

    operating_system = st.selectbox(
        "Operating System",
        list(range(1, 9))
    )

    browser = st.selectbox(
        "Browser",
        list(range(1, 14))
    )

    region = st.selectbox(
        "Region",
        list(range(1, 10))
    )

with col2:
    traffic_type = st.selectbox(
        "Traffic Type",
        list(range(1, 21))
    )

    visitor_type = st.selectbox(
        "Visitor Type",
        ["Returning_Visitor", "New_Visitor", "Other"]
    )

    weekend = st.selectbox(
        "Weekend",
        [False, True]
    )


st.divider()

if st.button(
    "Predict Purchase Intention",
    type="primary",
    use_container_width=True,
):

    input_data = pd.DataFrame(
        {
            "Administrative": [administrative],
            "Administrative_Duration": [administrative_duration],
            "Informational": [informational],
            "Informational_Duration": [informational_duration],
            "ProductRelated": [product_related],
            "ProductRelated_Duration": [product_related_duration],
            "BounceRates": [bounce_rates],
            "ExitRates": [exit_rates],
            "PageValues": [page_values],
            "SpecialDay": [special_day],
            "Month": [month],
            "OperatingSystems": [operating_system],
            "Browser": [browser],
            "Region": [region],
            "TrafficType": [traffic_type],
            "VisitorType": [visitor_type],
            "Weekend": [weekend],
        }
    )

    try:
        processed_input = preprocessor.transform(input_data)

        probability = float(
            model.predict(processed_input, verbose=0)[0][0]
        )

        prediction = int(probability >= threshold)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Purchase Likely")
        else:
            st.warning("Purchase Unlikely")

        st.metric(
            "Purchase Probability",
            f"{probability * 100:.2f}%"
        )

        st.caption(
            f"Classification Threshold: {threshold:.2f}"
        )

    except Exception as error:
        st.error(
            f"Prediction could not be generated: {error}"
        )
