# streamlitcode.py
import streamlit as st
import sys
import subprocess
import importlib
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import joblib

# ---------------------------
# AUTO-INSTALL PACKAGES
# ---------------------------
REQUIRED_PACKAGES = ["pandas","numpy","scikit-learn","joblib","streamlit"]

def install_package(pkg):
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
                st.warning(f"Could not install {pkg}: {e}")

ensure_packages(REQUIRED_PACKAGES)

# ---------------------------
# DIRECTORIES
# ---------------------------
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------
# 1. DATA GENERATION
# ---------------------------
def generate_dummy_data(path=DATA_DIR / "tourism_data.csv"):
    destinations = [
        ("Goa","beach"),("Manali","mountain"),("Jaipur","culture"),
        ("Varanasi","spiritual"),("Kerala","backwater"),("Rishikesh","adventure")
    ]
    users = list(range(100, 200))
    rows = []
    rng = np.random.default_rng(42)
    reviews = [
        "Great place with excellent beaches.",
        "Too crowded during holidays.",
        "Peaceful and beautiful scenery.",
        "Good food and hotels.",
        "Amazing views, perfect for families.",
        "Not well maintained in some areas.",
        "Must visit for culture lovers.",
        "Excellent hospitality.",
        "Adventure sports were thrilling.",
        "Budget-friendly and great food."
    ]
    for year in [2021,2022,2023]:
        for month in range(1,13):
            for dest,dtype in destinations:
                base = {
                    "Goa":   5000 if month in [11,12,1,2] else 1500 if month in [6,7,8] else 3000,
                    "Manali":4500 if month in [5,6,7,8] else 3000 if month in [12,1,2] else 2000,
                    "Jaipur":4000 if month in [11,12,1] else 2500,
                    "Varanasi":2800,
                    "Kerala":4200 if month in [11,12,1,2] else 2600,
                    "Rishikesh":3200 if month in [5,6,9] else 2100
                }.get(dest,2000)
                visitors = int(max(500, base + rng.normal(0,300)))
                selected = rng.choice(users,size=5,replace=False)
                for u in selected:
                    rating = round(float(max(1,min(5,rng.normal(4.0,0.5)))),1)
                    review = random.choice(reviews)
                    rows.append({
                        "user_id": int(u),
                        "destination": dest,
                        "year": year,
                        "month": month,
                        "visitors": visitors,
                        "rating": rating,
                        "review_text": review,
                        "cost_level": 3 if dest in ["Goa","Kerala"] else 2,
                        "dest_type": dtype,
                        "city": dest
                    })
    df = pd.DataFrame(rows)
    df.to_csv(path,index=False)
    return df

def get_monthly(df):
    # Visitors aggregation only
    return df.groupby(["year","month","destination"])["visitors"].mean().reset_index()

# ---------------------------
# 2. MODELS
# ---------------------------
TF = None
TFIDF = None
DESTS = None

def tfidf_init(df):
    global TF, TFIDF, DESTS
    agg = df.groupby("destination")["review_text"].apply(" ".join).reset_index()
    TF = TfidfVectorizer(max_features=500)
    TFIDF = TF.fit_transform(agg["review_text"])
    DESTS = list(agg["destination"])

def content_rec(df,text):
    global TF, TFIDF, DESTS
    if TF is None:
        tfidf_init(df)
    vec = TF.transform([text])
    sims = cosine_similarity(vec,TFIDF).flatten()
    idx = np.argsort(-sims)[:5]
    return [DESTS[i] for i in idx]

def item_cf(user_id,df):
    pivot = df.pivot_table(index="user_id", columns="destination", values="rating", aggfunc="mean").fillna(0)
    if user_id not in pivot.index:
        return list(pivot.columns[:5])
    sim = cosine_similarity(pivot.T.values)
    u = pivot.loc[user_id].values
    scores = np.dot(u, sim)
    scores[u>0] = -np.inf
    idx = np.argsort(-scores)[:5]
    return list(pivot.columns[idx])

def hybrid(df, uid, text):
    a = item_cf(uid, df)
    b = content_rec(df, text)
    final = list(dict.fromkeys(a+b))
    return final[:5]

# ---------------------------
# FORECASTING
# ---------------------------
def prepare_features(df):
    df = df.copy()
    df["dest_code"] = df["destination"].astype("category").cat.codes
    df["s"] = np.sin(2*np.pi*df["month"]/12)
    df["c"] = np.cos(2*np.pi*df["month"]/12)
    X = df[["year","month","dest_code","s","c"]]
    y = df["visitors"]
    return X,y,df

def train_model(df):
    X,y,df = prepare_features(df)
    model = RandomForestRegressor(n_estimators=200)
    model.fit(X,y)
    joblib.dump(model,MODELS_DIR/"rf.pkl")
    return model, df

def predict(model, df, dest, year, month):
    cats = list(df["destination"].astype("category").cat.categories)
    code = cats.index(dest) if dest in cats else 0
    s = np.sin(2*np.pi*month/12)
    c = np.cos(2*np.pi*month/12)
    X = np.array([[year, month, code, s, c]])
    return int(model.predict(X)[0])

def overcrowd(df,dest,pred):
    d = df[df["destination"]==dest]["visitors"]
    mean = d.mean()
    sd = d.std() if d.std()>0 else 1
    z = (pred-mean)/sd
    return z>1.5, z

# ---------------------------
# ADVANCED FEATURES
# ---------------------------
ACTIVITIES = {
    "beach":["Sunbathing","Surfing","Beach Volleyball"],
    "mountain":["Trekking","Camping","Skiing"],
    "culture":["Museum Tours","Heritage Walks","Local Cuisine Tasting"],
    "spiritual":["Temple Visits","Meditation Retreats","River Ghats"],
    "backwater":["Houseboat Ride","Fishing","Village Tour"],
    "adventure":["Rafting","Bungee Jumping","Zip Lining"]
}
NEARBY = {
    "Goa":["Kerala","Mumbai"],
    "Manali":["Shimla","Kullu"],
    "Jaipur":["Udaipur","Jodhpur"],
    "Varanasi":["Allahabad","Bodh Gaya"],
    "Kerala":["Goa","Coorg"],
    "Rishikesh":["Haridwar","Dehradun"]
}

def dynamic_cost(dest,month,base_cost=2000):
    peak_months = [11,12,1,2] if dest in ["Goa","Kerala"] else [5,6,7,8] if dest=="Manali" else [11,12,1]
    multiplier = 1.5 if month in peak_months else 1.0
    return int(base_cost*multiplier)

def best_visit_month(df,dest):
    monthly_visitors = df[df["destination"]==dest].groupby("month")["visitors"].mean()
    best_month = monthly_visitors.idxmin()
    return best_month,int(monthly_visitors.min())

def suggest_activities(df,dest):
    dtype = df[df["destination"]==dest]["dest_type"].iloc[0]
    return ACTIVITIES.get(dtype,[])

def advanced_recommendation(df,profile):
    df_agg = df.groupby("destination").agg({"rating":"mean","visitors":"mean","cost_level":"mean"}).reset_index()
    df_agg["score"] = df_agg["rating"]/5 + (1/df_agg["visitors"])*100 + (1/df_agg["cost_level"])
    df_agg = df_agg.sort_values("score",ascending=False)
    return df_agg["destination"].head(5).tolist()

def predict_with_confidence(model,df,dest,year,month):
    pred = predict(model,df,dest,year,month)
    d = df[df["destination"]==dest]["visitors"]
    std_dev = d.std() if d.std()>0 else 1
    lower = int(pred - 1.96*std_dev)
    upper = int(pred + 1.96*std_dev)
    return pred, lower, upper

def alternative_dest(dest,over):
    return NEARBY.get(dest,[]) if over else []

def cost_suggestion(df,dest):
    return df[df["destination"]==dest]["cost_level"].mode().iloc[0]

def destination_summary(df,dest,pred,uid,profile,agg,model):
    ovr,z = overcrowd(agg,dest,pred)
    recs = hybrid(df,uid,profile)
    cost = cost_suggestion(df,dest)
    best_month,min_vis = best_visit_month(agg,dest)
    activities = suggest_activities(df,dest)
    adv_recs = advanced_recommendation(df,profile)
    pred_ci,low,high = predict_with_confidence(model,agg,dest,2025,11)
    alt = alternative_dest(dest,ovr)

    summary = {
        "Predicted visitors": pred,
        "95% CI": f"{low}-{high}",
        "Overcrowding": f"{ovr} (z-score={round(z,2)})",
        "Recommended budget": cost,
        "Best month": f"{best_month} (avg visitors={min_vis})",
        "Top Recommendations": recs,
        "Advanced Destinations": adv_recs,
        "Activities": activities,
        "Alternative Destinations": alt
    }
    return summary

# ---------------------------
# STREAMLIT INTERFACE
# ---------------------------
st.title("🌏 Smart Tourism AI System")

df = generate_dummy_data()
agg = get_monthly(df)
model, _ = train_model(agg)

dest = st.text_input("Enter Destination Name")
year = st.number_input("Year",2025,2030,2025)
month = st.number_input("Month (1-12)",1,12,11)
profile = st.text_area("Describe your preferences")
uid = 101  # Example user

if st.button("Get Recommendation"):
    pred = predict(model,agg,dest,year,month)
    summary = destination_summary(df,dest,pred,uid,profile,agg,model)

    for k,v in summary.items():
        st.write(f"**{k}:** {v}")
