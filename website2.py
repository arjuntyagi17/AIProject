import streamlit as st
import pyrebase
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from datetime import datetime

#import json
#import streamlit.secrets

firebaseConfig = {
    "apiKey": st.secrets["firebase"]["apiKey"],
    "authDomain": st.secrets["firebase"]["authDomain"],
    "databaseURL": st.secrets["firebase"]["databaseURL"],
    "projectId": st.secrets["firebase"]["projectId"],
    "storageBucket": st.secrets["firebase"]["storageBucket"],
    "messagingSenderId": st.secrets["firebase"]["messagingSenderId"],
    "appId": st.secrets["firebase"]["appId"],
    "measurementId": st.secrets["firebase"]["measurementId"],
}

# 🔥 Firebase Configuration
# firebaseConfig = {
#     "apiKey": "AIzaSyCJ5Fu8iQa_sUG3QA5PGwf9GhLyrIUcYEQ",
#     "authDomain": "arjun-tyagi.firebaseapp.com",
#     "databaseURL": "https://arjun-tyagi-default-rtdb.firebaseio.com/",
#     "projectId": "arjun-tyagi",
#     "storageBucket": "arjun-tyagi.appspot.com",
#     "messagingSenderId": "135681411695",
#     "appId": "1:135681411695:web:9e066226dec85cbae0db9a",
#     "measurementId": "G-1J691M74X9"
# }

# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

st.title("Image Uploader")
st.text("Upload an image to run the YOLO classifier.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Submit"):
        st.text("Running YOLO model...")

        try:
            # 🔹 Load YOLO model
            model = YOLO("best.pt")  
            results = model(image)  

            # 🔹 Extract predictions (only class name)
            predictions_df = results[0].to_df()  # Convert YOLO results to DataFrame
            predictions_list = predictions_df["name"].tolist()  # Extract only the "name" column

            # 🔥 **Fix: Convert List into Dictionary for Firebase**
            predictions_dict = {name: True for name in predictions_list}  # ✅ Store names as keys
            
            # 🔹 Format Firebase Data
            data = {
                "predictions": predictions_dict,  # ✅ No numbered keys, class names as keys
                "timestamp": datetime.utcnow().isoformat()
            }

            # 🔥 Push to Firebase
            db.child("image_predictions").push(data)

            st.success("Predictions saved to Firebase!")

            # 🔥 **Fixing Streamlit Display**
            if len(predictions_list) == 1:
                st.write(f"Detected: **{predictions_list[0]}**")  # ✅ Show only class name
            else:
                st.write("Detected:", ", ".join(predictions_list))  # ✅ Display names properly

        except Exception as e:
            st.error(f"An error occurred: {e}")
