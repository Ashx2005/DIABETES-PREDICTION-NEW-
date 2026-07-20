import re
import pandas as pd
import sqlite3
import joblib
import streamlit as st
import bcrypt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# Database Setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    email TEXT UNIQUE,
                    password TEXT,
                    role TEXT DEFAULT 'user')''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
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
                    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Password Hashing
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# Email Validation
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email)

# Password Strength Validation
def is_valid_password(password):
    pattern = r"^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
    return re.match(pattern, password)

# User Authentication
def register_user(username, email, password):
    if not is_valid_email(email):
        st.error("Invalid email format!")
        return False
    if not is_valid_password(password):
        st.error("Password must be strong!")
        return False
    
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hash_password(password)))
        conn.commit()
        st.success("Registration Successful! Please login.")
        return True
    except sqlite3.IntegrityError:
        st.error("Username or email already exists.")
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user and verify_password(password, user[1])

# Load and Train Model
df = pd.read_csv("diabetes.csv")
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Save Model
joblib.dump((model, scaler), "diabetes_model.pkl")

# Feature Importance Analysis
feature_importances = model.feature_importances_
feature_names = X.columns

# Streamlit App
init_db()
st.title("Diabetes Prediction System")

if not st.session_state.logged_in:
    st.subheader("Login or Register")
    option = st.radio("Select Option", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if option == "Register":
        email = st.text_input("Email")
        if st.button("Register"):
            register_user(username, email, password)
    else:
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid Credentials")
else:
    st.subheader(f"Welcome, {st.session_state.username}")
    
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    BMI = st.number_input("BMI", min_value=0.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)
    
    if st.button("Predict"):
        model, scaler = joblib.load("diabetes_model.pkl")
        inputs = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, BMI, diabetes_pedigree, age]])
        inputs_scaled = scaler.transform(inputs)
        prediction = model.predict(inputs_scaled)
        confidence = model.predict_proba(inputs_scaled)[0]
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        confidence_score = max(confidence) * 100  # Convert to percentage
        
        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {confidence_score:.2f}%")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()
    
    # Display Feature Importance
    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importances, y=feature_names, ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance in Diabetes Prediction Model")
    st.pyplot(fig)