import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Multi-page Labs App",
    page_icon="📚",
    layout="wide"
)

# Create navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a Lab:", ["Lab 2 (Default)", "Lab 1"])

# Import and run the selected page
if page == "Lab 1":
    import Lab1
    Lab1.run()
elif page == "Lab 2 (Default)":
    import Lab2
    Lab2.run()
