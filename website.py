import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np

with st.sidebar:
    selected = option_menu(
        menu_title = None,
        options = ["Home", "Your Kitchen", "Settings", "Log in"],
        icons = ["house", "graph-up", "gear", "person"]
    )

if selected == "Home":
    st.title("Home")
    st.text("At [Name], we are focused on using artificial intelligence to combat food wastage in the restaurant industry. Our solution leverages advanced computer vision and machine learning to monitor food waste in real-time, providing restaurants with the tools to optimize their operations, reduce waste, and improve sustainability.\n\n")
    st.image("website/kitchen1.png")
    st.text("We deploy a camera system placed above kitchen dustbins to capture and analyze discarded food. Our AI model accurately identifies the type and quantity of waste, categorizing it by ingredient and reason for disposal. This data is then presented through a comprehensive dashboard, offering restaurant owners actionable insights into food waste patterns and inefficiencies.")
    st.image("website/kitchen2.png", width=900)
    st.text(" \n\nBy leveraging this data, restaurant owners can adjust inventory levels, refine portion sizes, and better align food supply with actual demand. The result is a more efficient operation, reduced waste, and lower costs—all while contributing to environmental sustainability. \n\nOur goal is to empower restaurants with the tools they need to minimize food waste, streamline their supply chain, and operate more responsibly.")

elif selected == "Your Kitchen":
    st.title("Your Kitchen")
    df = pd.DataFrame(np.random.rand(50, 20), columns=("col %d" % i for i in range(20)))

    st.text("")
    st.text("")
    st.line_chart(df)
    st.text("")
    st.text("")
    st.area_chart(df)
    st.text("")
    st.text("")
    st.bar_chart(df)