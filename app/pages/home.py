import streamlit as st

st.title("Online Shopper Purchase Intention Prediction")
st.subheader("ANN Classification Project")

st.write(
    """
    This project uses an Artificial Neural Network (ANN) classification
    model to predict whether an online shopping session will result in
    a purchase.
    """
)

st.divider()

st.header("Project Overview")
st.write(
    """
    Online shoppers interact with e-commerce websites in different ways.
    Their browsing behaviour, page visits, session duration, bounce rates,
    page values, visitor type, and other session characteristics can provide
    useful information about their purchase intention.

    The objective of this project is to analyze online shopping session
    data and build an ANN model that predicts whether a visitor is likely
    to complete a purchase.
    """
)

st.divider()

st.header("Dataset Information")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", "12,205")
col2.metric("Columns", "18")
col3.metric("Target", "Revenue")

st.write(
    """
    After data cleaning, the dataset contains **12,205 records and
    18 columns** with no missing values.

    The target variable, **Revenue**, indicates whether an online shopping
    session resulted in a purchase.
    """
)

st.divider()

st.header("Target Variable")
st.markdown(
    """
    - **True / 1:** The shopping session resulted in a purchase.
    - **False / 0:** The shopping session did not result in a purchase.
    """
)

st.divider()

st.header("Input Features")
st.write(
    """
    The ANN model uses information related to the visitor's browsing session.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        - Administrative
        - Administrative Duration
        - Informational
        - Informational Duration
        - Product Related
        - Product Related Duration
        - Bounce Rates
        - Exit Rates
        - Page Values
        """
    )

with col2:
    st.markdown(
        """
        - Special Day
        - Month
        - Operating Systems
        - Browser
        - Region
        - Traffic Type
        - Visitor Type
        - Weekend
        """
    )

st.divider()

st.header("ANN Classification Model")
st.write(
    """
    An **Artificial Neural Network (ANN)** was developed to classify
    online shopping sessions into purchase and non-purchase classes.

    Different techniques were evaluated to improve classification
    performance and address class imbalance. The final ANN produces
    a purchase probability, which is converted into a class prediction
    using the selected classification threshold.
    """
)

st.divider()

st.header("Application Pages")
st.markdown(
    """
    - **Home:** Project and dataset overview.
    - **Analysis:** Explore patterns and visualizations from the cleaned dataset.
    - **Model:** Enter shopping-session details and predict purchase intention.
    """
)

st.info("Choose Analysis or Model from the sidebar to explore the application.")
