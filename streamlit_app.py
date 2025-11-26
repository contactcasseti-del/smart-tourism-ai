# ================================================
# STREAMLIT VERSION OF YOUR AI TOURISM SYSTEM
# (NO CHANGES TO YOUR ORIGINAL LOGIC)
# EXTRA FEATURES, UI, COLORS & ANIMATIONS ADDED
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
from pathlib import Path
from datetime import datetime

# Importing everything from your original code file
# (PASTE YOUR WHOLE CODE BELOW — WITHOUT EDITING)
# -----------------------------------------------
# START OF ORIGINAL CODE
# -----------------------------------------------

import sys
import subprocess
import importlib
import os
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import joblib

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "joblib",
]

def install_package(pkg):
    print(f"[SETUP] Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def ensure_packages(pkgs):
    for pkg in pkgs:
        try:
            if importlib.util.find_spec(pkg) is None:
                install_package(pkg)
        except:
            try:
                install_package(pkg)
            except Exception as e:
                print(f"[ERROR] Could not install {pkg}: {e}")

ensure_packages(REQUIRED_PACKAGES)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def generate_dummy_data(path=DATA_DIR / "tourism_data.csv"):
    destinations = [
        ("Goa","beach"),("Manali","mountain"),("Jaipur","culture"),
        ("Varanasi","spiritual"),("Kerala","backwater"),("Rishikesh","adventure")
    ]

    users = list(range(100, 200))
    rows = []
    rng = np.random.default_rng(42)

    reviews = [
        "Great place with excellent beaches.","Too crowded during holidays.",
        "Peaceful and beautiful scenery.","Good food and hotels.",
        "Amazing views, perfect for families.","Not well maintained in some areas.",
        "Must visit for culture lovers.","Excellent hospitality.",
        "Adventure sports were thrilling.","Budget-friendly and great food."
    ]

    for year in [2021, 2022, 2023]:
        for month in range(1, 13):
            for dest, dtype in destinations:

                base = {
                    "Goa":   5000 if month in [11,12,1,2] else 1500 if month in [6,7,8] else 3000,
                    "Manali":4500 if month in [5,6,7,8] else 3000 if month in [12,1,2] else 2000,
                    "Jaipur":4000 if month in [11,12,1] else 2500,
                    "Varanasi":2800,
                    "Kerala":4200 if month in [11,12,1,2] else 2600,
                    "Rishikesh":3200 if month in [5,6,9] else 2100
                }.get(dest, 2000)

                visitors = int(max(500, base + rng.normal(0, 300)))
                selected = rng.choice(users, size=5, replace=False)

                for u in selected:
                    rating = round(float(max(1, min(5, rng.normal(4.0, 0.5)))),1)
                    review = random.choice(reviews)

                    rows.append({
                        "user_id": int(u),"destination": dest,"year": year,
                        "month": month,"visitors": visitors,"rating": rating,
                        "review_text": review,"cost_level": 3 if dest in ["Goa","Kerala"] else 2,
                        "dest_type": dtype,"city": dest
                    })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

def load_data():
    df = pd.read_csv(DATA_DIR/"tourism_data.csv")
    df["review_text"] = df["review_text"].fillna("")
    return df

def get_monthly(df):
    return df.groupby(["year","month","destination"])["visitors"].mean().reset_index()

TF = None
TFIDF = None
DESTS = None

def tfidf_init(df):
    global TF, TFIDF, DESTS
    agg = df.groupby("destination")["review_text"].apply(" ".join).reset_index()
    TF = TfidfVectorizer(max_features=500)
    TFIDF = TF.fit_transform(agg["review_text"])
    DESTS = list(agg["destination"])

def content_rec(df, text):
    global TF, TFIDF, DESTS
    if TF is None:
        tfidf_init(df)
    vec = TF.transform([text])
    sims = cosine_similarity(vec, TFIDF).flatten()
    idx = np.argsort(-sims)[:5]
    return [DESTS[i] for i in idx]

def item_cf(user_id, df):
    pivot = df.pivot_table(index="user_id", columns="destination", values="rating", aggfunc="mean").fillna(0)
    if user_id not in pivot.index:
        return list(pivot.columns[:5])
    sim = cosine_similarity(pivot.T.values)
    u = pivot.loc[user_id].values
    scores = np.dot(u, sim)
    scores[u > 0] = -np.inf
    idx = np.argsort(-scores)[:5]
    return list(pivot.columns[idx])

def hybrid(df, uid, text):
    a = item_cf(uid, df)
    b = content_rec(df, text)
    final = list(dict.fromkeys(a + b))
    return final[:5]

def prepare_features(df):
    df = df.copy()
    df["dest_code"] = df["destination"].astype("category").cat.codes
    df["s"] = np.sin(2*np.pi*df["month"]/12)
    df["c"] = np.cos(2*np.pi*df["month"]/12)
    X = df[["year","month","dest_code","s","c"]]
    y = df["visitors"]
    return X, y, df

def train_model(df):
    X, y, df = prepare_features(df)
    model = RandomForestRegressor(n_estimators=200)
    model.fit(X, y)
    joblib.dump(model, MODELS_DIR/"rf.pkl")
    return model, df

def predict(model, df, dest, year, month):
    cats = list(df["destination"].astype("category").cat.categories)
    code = cats.index(dest) if dest in cats else 0
    s = np.sin(2*np.pi*month/12)
    c = np.cos(2*np.pi*month/12)
    X = np.array([[year, month, code, s, c]])
    return int(model.predict(X)[0])

def overcrowd(df, dest, pred):
    d = df[df["destination"] == dest]["visitors"]
    mean = d.mean()
    sd = d.std() if d.std() > 0 else 1
    z = (pred - mean) / sd
    return z > 1.5, z

# -----------------------------------------------
# END OF ORIGINAL CODE
# -----------------------------------------------


# ===========================================================
# STREAMLIT UI STARTS HERE
# ===========================================================

st.set_page_config(page_title="AI Smart Tourism System",
                   page_icon="🌍",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .main {background-color: #f0f6ff;}
    .title {color:#0047ab; text-align:center; font-size:50px; 
            font-weight:bold; margin-bottom:20px;}
    .subheading {color:#0080ff; font-size:22px; text-align:center;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>🌍 SMART TOURISM AI SYSTEM</div>", unsafe_allow_html=True)
st.markdown("<div class='subheading'>AI-powered forecasting • Recommendations • Overcrowding Alerts</div>", unsafe_allow_html=True)

st.write("")

# --- Load & Prepare Data ---
with st.spinner("Generating dataset..."):
    df = generate_dummy_data()
    agg = get_monthly(df)
    model, df2 = train_model(agg)

st.success("Dataset & Model Ready!")

# Sidebar
st.sidebar.header("User Input")
dest = st.sidebar.selectbox("Destination", sorted(df["destination"].unique()))
year = st.sidebar.number_input("Year", min_value=2024, max_value=2035, value=2025)
month = st.sidebar.slider("Month", 1, 12, 5)
profile = st.sidebar.text_area("Describe your travel preferences:")

run_btn = st.sidebar.button("Run Prediction 🚀")

# --- WHEN RUN BUTTON IS CLICKED ---
if run_btn:

    with st.spinner("Processing your request..."):
        predicted_visitors = predict(model, agg, dest, year, month)
        over, z_val = overcrowd(agg, dest, predicted_visitors)
        recommendations = hybrid(df, 101, profile)

    st.success("Prediction Complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Visitor Forecast")
        st.metric(label="Predicted Visitors", value=predicted_visitors)

        st.subheader("🚨 Overcrowding Risk")
        risk = "High ❗" if over else "Low ✅"
        st.metric(label="Risk Level", value=risk, delta=round(z_val, 2))

    with col2:
        st.subheader("🎯 Top Recommended Destinations")
        st.write("Based on your profile + AI models:")

        st.markdown(
            "<ul style='font-size:20px; color:#0047ab;'>"
            + "".join([f"<li>{rec}</li>" for rec in recommendations])
            + "</ul>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("📈 Monthly Visitor Patterns")
    chart_df = agg[agg["destination"] == dest]
    st.line_chart(chart_df[["month", "visitors"]].set_index("month"))

else:
    st.info("Enter your details in sidebar and click **Run Prediction**")

st.markdown("---")
st.markdown("### 🌈 Built with Streamlit • Machine Learning • Love ❤️")

