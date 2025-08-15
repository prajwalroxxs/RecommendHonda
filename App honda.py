import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import base64

# Load dataset
df = pd.read_csv("honda_india_current_models_2025.csv")
df.fillna("", inplace=True)

# Normalize numeric columns
numeric_cols = ["Price", "Mileage", "Seating", "Engine_CC"]
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Create combined features for content-based filtering
df["combined_features"] = df["Engine_Type"].astype(str) + " " + df["Features"].astype(str) + " " + df["Body_Type"].astype(str)
count_matrix = CountVectorizer().fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

# Recommendation function
def recommend_cars(budget, seating, mileage, engine_type):
    budget_scaled = scaler.transform([[budget, mileage, seating, 0]])[0][0]
    filtered_df = df[
        (df["Price"] <= budget_scaled) &
        (df["Seating"] >= seating/10) &
        (df["Engine_Type"].str.contains(engine_type, case=False))
    ]
    if filtered_df.empty:
        return pd.DataFrame(columns=df.columns)
    filtered_df["score"] = (
        filtered_df["Mileage"] * 0.4 +
        filtered_df["Price"] * 0.3 +
        filtered_df["Seating"] * 0.2 +
        np.random.rand(len(filtered_df)) * 0.1
    )
    return filtered_df.sort_values(by="score", ascending=False).head(5)

# Background image
def set_bg(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Honda Car Recommendation System", layout="wide")
set_bg("3bde3e6d-da82-4d91-9a41-aca7bb9bd133.png")
st.title("🚗 Honda Car Recommendation System")
st.markdown("### Find the perfect Honda model for you!")

# Sidebar form
st.sidebar.header("Your Preferences")
budget = st.sidebar.number_input("Budget (in Lakh ₹)", min_value=5, max_value=50, value=15)
seating = st.sidebar.selectbox("Seating Capacity", [4, 5, 7])
mileage = st.sidebar.number_input("Minimum Mileage (km/l)", min_value=10, max_value=30, value=15)
engine_type = st.sidebar.selectbox("Preferred Engine Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Any"])

if st.sidebar.button("Recommend"):
    results = recommend_cars(budget, seating, mileage, engine_type if engine_type != "Any" else "")
    if results.empty:
        st.warning("No matching cars found. Try adjusting your preferences.")
    else:
        for _, row in results.iterrows():
            st.markdown(f"## {row['Model_Name']}")
            st.write(f"**Price:** ₹{row['Price']*50:.2f} Lakh")
            st.write(f"**Mileage:** {row['Mileage']*30:.1f} km/l")
            st.write(f"**Seating:** {int(row['Seating']*10)}")
            st.write(f"**Engine Type:** {row['Engine_Type']}")
            st.write(f"**Features:** {row['Features']}")
            st.write("---")
