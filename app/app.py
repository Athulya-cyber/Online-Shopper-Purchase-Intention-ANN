import streamlit as st

st.set_page_config(
    page_title="Online Shopper Purchase Intention",
    page_icon="🛒",
    layout="wide"
)

pages = [
    st.Page("Pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("Pages/analysis.py", title="Analysis", icon="📊"),
    st.Page("Pages/model.py", title="Model", icon="🤖"),
]

navigation = st.navigation(pages)
navigation.run()
