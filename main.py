
# NOTE:
# This version loads an already-trained diabetes_model.pkl instead of
# retraining the model every time the app starts.

import re
import sqlite3
import joblib
import bcrypt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Session State
# --------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --------------------------
# Database
# --------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT DEFAULT 'user'
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        pregnancies INTEGER,
        glucose INTEGER,
        blood_pressure INTEGER,
        skin_thickness INTEGER,
        insulin INTEGER,
        BMI REAL,
        diabetes_pedigree REAL,
        age INTEGER,
        result TEXT,
        confidence REAL,
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

# --------------------------
# Password
# --------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# --------------------------
# Validation
# --------------------------
def is_valid_email(email):
    return re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email)

def is_valid_password(password):
    return re.match(r"^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$", password)

# --------------------------
# Authentication
# --------------------------
def register_user(username,email,password):

    if not is_valid_email(email):
        st.error("Invalid email format!")
        return

    if not is_valid_password(password):
        st.error("Password must contain:\n- 8+ characters\n- 1 uppercase\n- 1 number\n- 1 special character")
        return

    conn=sqlite3.connect("users.db")
    c=conn.cursor()

    try:
        c.execute(
            "INSERT INTO users(username,email,password) VALUES(?,?,?)",
            (username,email,hash_password(password))
        )
        conn.commit()
        st.success("Registration Successful!")
    except sqlite3.IntegrityError:
        st.error("Username or Email already exists.")
    finally:
        conn.close()

def authenticate_user(username,password):

    conn=sqlite3.connect("users.db")
    c=conn.cursor()

    c.execute(
        "SELECT password FROM users WHERE username=?",
        (username,)
    )

    user=c.fetchone()
    conn.close()

    if user:
        return verify_password(password,user[0])

    return False

# --------------------------
# Load Model
# --------------------------
model, scaler = joblib.load("diabetes_model.pkl")

feature_names = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "Diabetes Pedigree",
    "Age"
]

feature_importances = model.feature_importances_

# --------------------------
# App
# --------------------------
init_db()

st.title("Diabetes Prediction System")

if not st.session_state.logged_in:

    st.subheader("Login / Register")

    option = st.radio(
        "Choose",
        ["Login","Register"]
    )

    username = st.text_input("Username")
    password = st.text_input("Password",type="password")

    if option=="Register":

        email=st.text_input("Email")

        if st.button("Register"):
            register_user(username,email,password)

    else:

        if st.button("Login"):

            if authenticate_user(username,password):
                st.session_state.logged_in=True
                st.session_state.username=username
                st.rerun()
            else:
                st.error("Invalid Credentials")

else:

    st.success(f"Welcome {st.session_state.username}")

    pregnancies = st.number_input("Pregnancies",0)
    glucose = st.number_input("Glucose",0)
    blood_pressure = st.number_input("Blood Pressure",0)
    skin_thickness = st.number_input("Skin Thickness",0)
    insulin = st.number_input("Insulin",0)
    BMI = st.number_input("BMI",0.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function",0.0)
    age = st.number_input("Age",0)

    if st.button("Predict"):

        data=np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            BMI,
            diabetes_pedigree,
            age
        ]])

        data=scaler.transform(data)

        prediction=model.predict(data)[0]
        probability=model.predict_proba(data)[0]

        result="Diabetic" if prediction==1 else "Non-Diabetic"
        confidence=max(probability)*100

        st.success(f"Prediction : {result}")
        st.info(f"Confidence : {confidence:.2f}%")

    if st.button("Logout"):
        st.session_state.logged_in=False
        st.session_state.username=""
        st.rerun()

    st.subheader("Feature Importance")

    fig,ax=plt.subplots(figsize=(8,5))

    sns.barplot(
        x=feature_importances,
        y=feature_names,
        ax=ax
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    st.pyplot(fig)
