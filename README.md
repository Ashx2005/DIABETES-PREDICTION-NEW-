# DIAPREDICT-ML-Auth 🩺🔍  

A **Machine Learning-powered Diabetes Prediction Model** with authentication, built using Python and Streamlit.

## 🚀 Features  
- Predicts the likelihood of diabetes based on input parameters  
- Uses a trained machine learning model (`diabetes_model.pkl`)  
- Web-based interface using **Streamlit**  
- Authentication system for user access  

## 📂 Project Structure  
```
📦 Diabetes-Prediction-Main  
 ┣ 📜 check_db.py            # Handles database-related operations  
 ┣ 📜 main.py                # Streamlit app main file  
 ┣ 📜 diabetes.csv           # Dataset used for training  
 ┣ 📜 diabetes_model.pkl     # Trained ML model  
 ┣ 📜 users.db               # User authentication database  
 ┣ 📜 modules.txt            # Required Python packages  
 ┣ 📜 streamlit.txt          # Streamlit dependencies  
 ┣ 📜 .gitignore             # Files to ignore in Git  
 ┣ 📜 README.md              # Project documentation  
```

## 🛠 Installation & Usage  

### 1️⃣ Install Dependencies  
```bash
pip install -r modules.txt
pip install -r streamlit.txt
```

### 2️⃣ Run the Application  
```bash
streamlit run main.py
```

## 📊 Model Details  
- **Algorithm Used:** (Specify ML algorithm e.g., Logistic Regression, Random Forest, etc.)  
- **Accuracy:** (Mention model accuracy if available)  
- **Dataset:** PIMA Diabetes Dataset (or specify source)  

## 🔐 Authentication System  
- Users need to log in before using the prediction model.  
- User details are stored in `users.db`.  

## 📌 Future Improvements  
- Improve model accuracy with feature engineering  
- Add more authentication options (Google OAuth, etc.)  
- Deploy the app using **Heroku** or **Render**  

---

### 🔗 Connect with Me  
🌐 GitHub: [Ashx2005](https://github.com/Ashx2005)  
🚀 Happy Coding!  
