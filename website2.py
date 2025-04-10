import streamlit as st
import pyrebase
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from PIL import Image

# -----------------------------
# Firebase Configuration
# -----------------------------
firebaseConfig = {
    "apiKey": "AIzaSyCJ5Fu8iQa_sUG3QA5PGwf9GhLyrIUcYEQ",
    "authDomain": "arjun-tyagi.firebaseapp.com",
    "databaseURL": "https://arjun-tyagi-default-rtdb.firebaseio.com/",
    "projectId": "arjun-tyagi",
    "storageBucket": "arjun-tyagi.appspot.com",
    "messagingSenderId": "135681411695",
    "appId": "1:135681411695:web:9e066226dec85cbae0db9a",
    "measurementId": "G-1J691M74X9"
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio("Select Page", ["Upload Image", "View Statistics"])

# -----------------------------
# Upload Image Page
# -----------------------------
if page == "Upload Image":
    st.title("Image Uploader")
    st.write("Upload an image to run the YOLO classifier.")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Submit"):
            st.write("Running YOLO model...")
            try:
                from ultralytics import YOLO
                model = YOLO("best.pt")
                results = model(image)
                predictions_df = results[0].to_df()
                predictions_list = predictions_df["name"].tolist()
                for prediction in predictions_list:
                    parts = prediction.split("_")
                    if len(parts) == 3:
                        grain, container, quantity = parts[0], parts[1], int(parts[2])
                        # Use current UTC timestamp when saving
                        data = {
                            "grain": grain,
                            "container": container,
                            "quantity": quantity,
                            "timestamp": datetime.utcnow().isoformat()  # e.g. "2024-03-01T10:32:36.970239"
                        }
                        db.child("image_predictions").push(data)
                st.success("Predictions saved to Firebase!")
                if len(predictions_list) == 1:
                    st.write(f"Detected: **{predictions_list[0]}**")
                else:
                    st.write("Detected:", ", ".join(predictions_list))
            except Exception as e:
                st.error(f"An error occurred: {e}")

# -----------------------------
# View Statistics Page
# -----------------------------
elif page == "View Statistics":
    st.title("Food Detection Statistics")
    data = db.child("image_predictions").get().val()
    if not data:
        st.warning("No data available.")
    else:
        # Process Firebase records treating the timestamp as a string.
        records = []
        for key, item in data.items():
            try:
                ts = item["timestamp"]  # e.g. "2024-03-01T10:32:36.970239"
                year = int(ts[0:4])
                month = int(ts[5:7])
                day = int(ts[8:10])
                date_obj = datetime(year, month, day)
                records.append({
                    "timestamp_str": ts,
                    "date": date_obj,
                    "grain": item["grain"],
                    "container": item["container"],
                    "quantity": item["quantity"]
                })
            except Exception as e:
                st.warning(f"Skipping invalid entry {key}: {e}")
        df = pd.DataFrame(records)
        if df.empty:
            st.warning("No valid data found in Firebase.")
            st.stop()

        # Collect unique years and grains for filter options.
        available_years = sorted(df["date"].dt.year.unique())
        available_grains = sorted(df["grain"].unique())

        # -----------------------------
        # All Filters in Sidebar Form
        # -----------------------------
        with st.sidebar.form("filters_form"):
            st.subheader("Select Filters")
            selected_year = st.selectbox("Select Year", available_years)
            selected_month = st.selectbox("Select Month", ["None"] + [str(i) for i in range(1, 13)])
            selected_week = st.selectbox("Select Week", ["None", "1", "2", "3", "4"])
            selected_grain = st.selectbox("Select Grain", available_grains)
            apply_button = st.form_submit_button("Apply Filters")

        if apply_button:
            # -----------------------------
            # Apply Year, Month, Week filters
            # -----------------------------
            filtered_df = df[df["date"].dt.year == selected_year].copy()
            if selected_month != "None":
                filtered_df = filtered_df[filtered_df["date"].dt.month == int(selected_month)]
            if selected_week != "None":
                w = int(selected_week)
                if w == 1:
                    filtered_df = filtered_df[(filtered_df["date"].dt.day >= 1) & (filtered_df["date"].dt.day <= 7)]
                elif w == 2:
                    filtered_df = filtered_df[(filtered_df["date"].dt.day >= 8) & (filtered_df["date"].dt.day <= 14)]
                elif w == 3:
                    filtered_df = filtered_df[(filtered_df["date"].dt.day >= 15) & (filtered_df["date"].dt.day <= 22)]
                elif w == 4:
                    filtered_df = filtered_df[filtered_df["date"].dt.day >= 23]

            if filtered_df.empty:
                st.warning("No data available for the selected Year/Month/Week.")
                st.stop()

            # -----------------------------
            # PIE CHART: Aggregate by Grain
            # -----------------------------
            st.write("### Grain Contribution (Pie Chart)")
            pie_df = filtered_df.groupby("grain")["quantity"].sum().reset_index()
            fig_pie = px.pie(pie_df, names="grain", values="quantity",
                             title="Grain Contribution")
            st.plotly_chart(fig_pie, use_container_width=True)

            # -----------------------------
            # TREND BAR CHART for Selected Grain
            # -----------------------------
            df_grain = filtered_df[filtered_df["grain"] == selected_grain].copy()
            if df_grain.empty:
                st.warning(f"No data available for grain: {selected_grain}")
                st.stop()

            # When only year is selected (Month == "None"), group by month (complete with zeros)
            if selected_month == "None":
                complete_months = pd.DataFrame({
                    "month": list(range(1, 13))
                })
                complete_months["month_label"] = complete_months["month"].apply(lambda m: datetime(2000, m, 1).strftime("%b"))
                trend_df = df_grain.groupby(df_grain["date"].dt.month)["quantity"].sum().reset_index()
                trend_df.columns = ["month", "quantity"]
                trend_df = pd.merge(complete_months, trend_df, on="month", how="left").fillna(0)
                x_axis = "month_label"
                x_axis_label = "Month"
            else:
                # When a month is selected, group by day, and display complete days with day-of-week labels.
                sel_year = selected_year
                sel_month = int(selected_month)
                # Determine the number of days in the selected month.
                if sel_month == 12:
                    next_month = datetime(sel_year + 1, 1, 1)
                else:
                    next_month = datetime(sel_year, sel_month + 1, 1)
                start_date_month = datetime(sel_year, sel_month, 1)
                days_in_month = (next_month - start_date_month).days

                complete_days = pd.DataFrame({
                    "day": list(range(1, days_in_month + 1))
                })
                # Create a label combining day-of-week and day number
                complete_days["date"] = complete_days["day"].apply(lambda d: datetime(sel_year, sel_month, d))
                complete_days["day_label"] = complete_days["date"].apply(lambda d: d.strftime("%a %d"))
                trend_df = df_grain.groupby(df_grain["date"].dt.day)["quantity"].sum().reset_index()
                trend_df.columns = ["day", "quantity"]
                trend_df = pd.merge(complete_days[["day", "day_label"]], trend_df, on="day", how="left").fillna(0)
                x_axis = "day_label"
                x_axis_label = "Day"

            st.write("### Grain Trend (Bar Chart)")
            fig_trend = px.bar(trend_df, x=x_axis, y="quantity",
                               title=f"{selected_grain} Trend",
                               labels={x_axis: x_axis_label, "quantity": "Total Quantity"})
            st.plotly_chart(fig_trend, use_container_width=True)

            # -----------------------------
            # WASTED QUANTITY CHART
            # -----------------------------
            st.write("### Wasted Quantity")
            df_waste = df_grain.copy()
            # Using the actual quantity as the value, or you can adjust as needed.
            df_waste["wasted"] = df_waste["quantity"] # 10% wastage assumption
            waste_df = df_waste.groupby("container")["wasted"].sum().reset_index()
            fig_waste = px.bar(waste_df, x="container", y="wasted",
                               title=f"Wasted Quantity of {selected_grain}",
                               labels={"container": "Container", "wasted": "Wasted Quantity"})
            st.plotly_chart(fig_waste, use_container_width=True)
