from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


# Project root is the parent directory of the Pages folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "Data" / "online_shoppers_intention_cleaned.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("Data Analysis")
st.write(
    """
    Explore the cleaned Online Shoppers Intention dataset
    using different visualizations.
    """
)

st.divider()

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: {DATA_PATH}. "
        "Make sure the cleaned CSV is inside the Data folder."
    )
    st.stop()

data = load_data(str(DATA_PATH))

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{data.shape[0]:,}")
col2.metric("Columns", data.shape[1])
col3.metric("Missing Values", int(data.isnull().sum().sum()))

if st.checkbox("Show Dataset"):
    st.dataframe(data, use_container_width=True)

st.divider()

st.header("Explore the Dataset")

analysis_option = st.selectbox(
    "Select Analysis",
    [
        "Revenue Distribution",
        "Visitor Type",
        "Month",
        "Weekend",
        "Page Values",
        "Bounce Rate vs Exit Rate",
    ],
)

if analysis_option == "Revenue Distribution":

    st.subheader("Purchase Distribution")

    revenue_counts = data["Revenue"].value_counts()
    total = len(data)

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(data=data, x="Revenue", ax=ax)

    ax.set_title("Revenue Distribution")
    ax.set_xlabel("Purchase")
    ax.set_ylabel("Number of Sessions")

    st.pyplot(fig)
    plt.close(fig)

    no_purchase = int(revenue_counts.get(False, 0))
    purchase = int(revenue_counts.get(True, 0))

    col1, col2 = st.columns(2)

    col1.metric(
        "No Purchase",
        f"{no_purchase:,}",
        f"{(no_purchase / total) * 100:.2f}%" if total else "0%",
    )

    col2.metric(
        "Purchase",
        f"{purchase:,}",
        f"{(purchase / total) * 100:.2f}%" if total else "0%",
    )

    st.write(
        """
        The target distribution shows the number of online shopping
        sessions that resulted in a purchase and those that did not.
        """
    )

elif analysis_option == "Visitor Type":

    st.subheader("Revenue by Visitor Type")

    visitor_data = pd.crosstab(
        data["VisitorType"],
        data["Revenue"]
    )

    st.bar_chart(visitor_data)

    st.write(
        "This visualization compares purchase and non-purchase sessions "
        "across different visitor types."
    )

elif analysis_option == "Month":

    st.subheader("Revenue by Month")

    month_data = pd.crosstab(
        data["Month"],
        data["Revenue"]
    )

    st.bar_chart(month_data)

    st.write(
        "This chart shows how purchase behaviour varies across months."
    )

elif analysis_option == "Weekend":

    st.subheader("Revenue by Weekend")

    weekend_data = pd.crosstab(
        data["Weekend"],
        data["Revenue"]
    )

    st.bar_chart(weekend_data)

    st.write(
        "This analysis compares purchase behaviour during weekdays and weekends."
    )

elif analysis_option == "Page Values":

    st.subheader("Page Values by Revenue")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.boxplot(
        data=data,
        x="Revenue",
        y="PageValues",
        ax=ax
    )

    ax.set_title("Page Values vs Revenue")
    ax.set_xlabel("Purchase")
    ax.set_ylabel("Page Values")

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "This plot compares Page Values between sessions that resulted "
        "in a purchase and sessions that did not."
    )

elif analysis_option == "Bounce Rate vs Exit Rate":

    st.subheader("Bounce Rate vs Exit Rate")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.scatterplot(
        data=data,
        x="BounceRates",
        y="ExitRates",
        hue="Revenue",
        alpha=0.6,
        ax=ax,
    )

    ax.set_title("Bounce Rate vs Exit Rate")
    ax.set_xlabel("Bounce Rate")
    ax.set_ylabel("Exit Rate")

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "This scatter plot shows the relationship between Bounce Rate "
        "and Exit Rate for purchase and non-purchase sessions."
    )
