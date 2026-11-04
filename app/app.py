import streamlit as st

st.set_page_config(
    page_title="Online Shopper Purchase Intention",
    page_icon="🛒",
    layout="wide"
)

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/analysis.py", title="Analysis", icon="📊"),
    st.Page("pages/model.py", title="Model", icon="🤖"),
]

navigation = st.navigation(pages)
navigation.run()
