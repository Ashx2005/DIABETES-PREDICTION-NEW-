import pandas as pd
import sqlite3
import joblib
import streamlit as st
import bcrypt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
                    password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    pregnancies INTEGER,
                    glucose INTEGER,
                    blood_pressure INTEGER,
                    skin_thickness INTEGER,
                    insulin INTEGER,
                    BMI REAL,
                    diabetes_pedigree REAL,
                    age INTEGER,
                    prediction TEXT)''')
    conn.commit()
    conn.close()

# Password Hashing
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# User Authentication
def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user and verify_password(password, user[0])

# Load Dataset and Train Model
df = pd.read_csv("diabetes.csv")
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "diabetes_model.pkl")

def store_prediction(username, inputs, prediction):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO predictions (username, pregnancies, glucose, blood_pressure, skin_thickness, insulin, BMI, diabetes_pedigree, age, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (username, *inputs, prediction))
    conn.commit()
    conn.close()

# Streamlit App
init_db()
st.title("Diabetes Prediction System with User Authentication")

# Debugging Session State
st.write("Session State:", st.session_state)

# If not logged in, show login/register page
if not st.session_state.logged_in:
    st.subheader("Login or Register")
    option = st.radio("Select Option", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        if st.button("Register"):
            if register_user(username, password):
                st.success("Registration Successful! Please login.")
            else:
                st.error("Username already exists.")
    else:  # Login
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()  # Force refresh after login
            else:
                st.error("Invalid Credentials")
else:
    # If logged in, show the prediction UI
    st.subheader(f"Welcome, {st.session_state.username}")
    
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    BMI = st.number_input("BMI", min_value=0.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)
    
    if st.button("Predict"):
        model = joblib.load("diabetes_model.pkl")
        inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, BMI, diabetes_pedigree, age]
        prediction = model.predict([inputs])
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        store_prediction(st.session_state.username, inputs, result)
        st.write(f"Prediction: {result}")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()  # Refresh the app to show login page
