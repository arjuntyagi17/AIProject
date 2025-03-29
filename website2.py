import streamlit as st
#from streamlit_option_menu import option_menu
#import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image

# with st.sidebar:
#     selected = option_menu(
#         menu_title=None,
#         options=["Home", "Your Kitchen", "Settings", "Image Uploader"],
#         icons=["house", "graph-up", "gear", "upload"]
#     )

# if selected == "Home":
#     st.title("Home")
#     st.text("At [Name], we are focused on using artificial intelligence to combat food wastage in the restaurant industry. Our solution leverages advanced computer vision and machine learning to monitor food waste in real-time, providing restaurants with the tools to optimize their operations, reduce waste, and improve sustainability.\n\n")
#     st.image("kitchen1.png")
#     st.text("We deploy a camera system placed above kitchen dustbins to capture and analyze discarded food. Our AI model accurately identifies the type and quantity of waste, categorizing it by ingredient and reason for disposal. This data is then presented through a comprehensive dashboard, offering restaurant owners actionable insights into food waste patterns and inefficiencies.")
#     st.image("kitchen2.png", width=900)
#     st.text(" \n\nBy leveraging this data, restaurant owners can adjust inventory levels, refine portion sizes, and better align food supply with actual demand. The result is a more efficient operation, reduced waste, and lower costs—all while contributing to environmental sustainability. \n\nOur goal is to empower restaurants with the tools they need to minimize food waste, streamline their supply chain, and operate more responsibly.")

# elif selected == "Your Kitchen":
#     st.title("Your Kitchen")
#     df = pd.DataFrame(np.random.rand(50, 20), columns=("col %d" % i for i in range(20)))
#     st.text("")
#     st.text("")
#     st.line_chart(df)
#     st.text("")
#     st.text("")
#     st.area_chart(df)
#     st.text("")
#     st.text("")
#     st.bar_chart(df)

# elif selected == "Settings":
#     st.title("Settings")

# elif selected == "Image Uploader":
#     st.title("Image Uploader")
#     st.text("Upload an image to run the YOLO classifier.")
#     uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#         if st.button("Submit"):
#             st.text("Running YOLO model...")
#             try:
#                  model = YOLO("best.pt")
#                  results = model(image)
#                  predictions = results.pandas().xyxy[0]
#                  st.write("Predictions:")
#                  st.write(predictions)
#             except Exception as e:
#                  st.error(f"An error occurred: {e}")



st.title("Image Uploader")
st.text("Upload an image to run the YOLO classifier.")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button("Submit"):
        st.text("Running YOLO model...")
        try:
             model = YOLO("best.pt")
             results = model(image)
             predictions = results[0].to_df()
             st.write("Predictions:")
             st.write(predictions)
        except Exception as e:
             st.error(f"An error occurred: {e}")
