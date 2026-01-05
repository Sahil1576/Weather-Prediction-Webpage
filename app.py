import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime

# Title
st.title("Seattle Weather Prediction App 🌦️")
st.markdown("Naive Bayes Classifier for Weather Prediction")

# Load model
model = joblib.load('LR_heart.pkl')  # તમે જે મોડલ સેવ કર્યો છે તે લોડ કરો

# Load data
df = pd.read_csv("seattle-weather.csv")

# Show raw data
with st.expander("📊 Show Raw Data"):
    st.write(df.head())

# Input form
st.subheader("🔍 Predict Weather")

# અહીં ફીચર પ્રમાણે ફોર્મ બનાવો (example માટે એક ફોર્મ આપેલું છે)
col1, col2 = st.columns(2)

input_date = st.date_input("Select Date", datetime.date.today())
input_day = input_date.day
input_month = input_date.month
input_year = input_date.year

with col1:
    temp_max = st.slider("Max Temperature (°C)", float(df['temp_max'].min()), float(df['temp_max'].max()), 20.0)
    temp_min = st.slider("Min Temperature (°C)", float(df['temp_min'].min()), float(df['temp_min'].max()), 10.0)

with col2:
    wind = st.slider("Wind Speed (mph)", float(df['wind'].min()), float(df['wind'].max()), 5.0)
    precipitation = st.slider("Precipitation (mm)", float(df['precipitation'].min()), float(df['precipitation'].max()), 0.0)


# Predict button
if st.button("📌 Predict Weather"):
    input_data = pd.DataFrame({
        'precipitation': [precipitation],
        'temp_max': [temp_max],
        'temp_min': [temp_min],
        'wind': [wind],
        'day':[input_day],
        'month':[input_month],
        'year':[input_year]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"🌤️ Predicted Weather: *{prediction}*")

# Optional: Visualization
st.subheader("📈 Data Visualization")

plot_option = st.selectbox("Select Chart Type", ["Histogram", "Correlation Heatmap", "Weather Frequency"])

if plot_option == "Histogram":
    col = st.selectbox("Select column", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

elif plot_option == "Correlation Heatmap":
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif plot_option == "Weather Frequency":
    fig, ax = plt.subplots()
    df['weather'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)
    
print(model.feature_names_in_)
print(df.select_dtypes(include=np.number).columns)