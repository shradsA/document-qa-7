import streamlit as st
import lab1
import lab2

# Sidebar navigation using st.navigation
selected = st.navigation(
    {
        "Labs": [
            st.Page(lab1.app, title="Lab 1", icon="📄"),
            st.Page(lab2.app, title="Lab 2", icon="🔬", default=True),  # Lab 2 is default
        ]
    }
)

# Run the selected page
selected.run()

