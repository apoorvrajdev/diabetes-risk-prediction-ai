# 🩺 Diabetes Risk Prediction AI

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Healthcare AI](https://img.shields.io/badge/Healthcare-AI-red)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project builds an **AI-powered clinical decision support system** that predicts the **risk of diabetes** using machine learning.

The system analyzes medical parameters such as:

- Age
- Hypertension
- Heart Disease
- Body Mass Index (BMI)
- HbA1c Level
- Blood Glucose Level
- Gender
- Smoking History

and predicts the **probability of diabetes risk**.

The model is deployed as an **interactive Streamlit web application** where users can input patient data and instantly receive a prediction.

---

# 🚀 Live Demo

[![Open App](https://img.shields.io/badge/Open%20Live%20App-Streamlit-red)](https://diabetes-risk-prediction-ai.streamlit.app)

OR Try the AI diabetes risk analyzer here:

https://diabetes-risk-prediction-ai.streamlit.app

Users can input medical parameters and instantly receive:

- Diabetes risk probability
- Visual risk meter
- Prediction confidence
- Medical interpretation

---

# 🧠 AI Prediction Interface

The application collects patient health information including age, BMI, blood glucose level, HbA1c, hypertension status, and smoking history to estimate diabetes risk.

---

## 🟢 Healthy Patient Prediction

### Patient Input Interface

*(Insert screenshot of healthy patient input page)*

### AI Prediction Result

*(Insert screenshot showing low-risk gauge)*

### Medical Interpretation

*(Insert screenshot showing healthy interpretation)*

---

## 🔴 High Risk Patient Prediction

### Patient Input Interface

*(Insert screenshot of high-risk patient inputs)*

### AI Prediction Result

*(Insert screenshot showing high-risk gauge)*

### Medical Interpretation

*(Insert screenshot showing high-risk interpretation)*

---

⚠️ **Disclaimer:**  
This AI system estimates diabetes risk and **does not replace professional medical diagnosis.**

---

# 📊 Model Performance

### Model Comparison

| Model | ROC-AUC Score |
|------|------|
| Logistic Regression | 0.96 |
| Decision Tree | 0.95 |
| Random Forest | **0.996** |
| Gradient Boosting | 0.97 |
| Gaussian Naive Bayes | 0.93 |

The **Random Forest model achieved the best performance** and was selected for deployment.

---

# 📈 Cross Validation

To ensure model robustness and reduce overfitting, **cross-validation techniques** were used during training.

Metrics evaluated:

- Accuracy
- Precision
- Recall
- ROC-AUC Score

The results show **consistently high performance across folds**, indicating strong generalization capability.

---

# 📂 Dataset

The dataset used is a **large diabetes prediction dataset used in healthcare machine learning research**.

Dataset characteristics:

- Number of samples: **100,000+**
- Number of features: **13 encoded input variables**
- Target variable: **Diabetes presence**

Each row represents a patient with medical attributes relevant to diabetes risk.

---

# ⚙️ Machine Learning Pipeline

The project follows a complete machine learning workflow:

1. Data loading using Pandas
2. Exploratory Data Analysis (EDA)
3. Data visualization using Seaborn and Matplotlib
4. Handling class imbalance using **SMOTE**
5. Train-test split
6. Hyperparameter tuning using **RandomizedSearchCV**
7. Model training
8. Model comparison
9. Model evaluation using ROC-AUC
10. Web application deployment using **Streamlit**

---

# 🧰 Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Jupyter Notebook

---

# 📁 Project Structure

```
diabetes-risk-prediction-ai
│
├── Prediction_Of_Diabetes.ipynb      # Model training notebook
├── diabetes_prediction_dataset.csv   # Dataset
├── streamlit_app.py                  # Streamlit web application
├── requirements.txt                  # Python dependencies
├── diabetes_model.pkl                # Trained ML model
├── .gitignore
└── README.md
```

---

# Installation

### Clone the repository

```
git clone https://github.com/apoorvrajdev/diabetes-risk-prediction-ai.git
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run the application

```
streamlit run streamlit_app.py
```

---

# 🚀 Future Improvements

Possible enhancements for the project:

- Use advanced models such as **XGBoost or LightGBM**
- Add **Explainable AI (SHAP)** for model transparency
- Deploy the system on **cloud infrastructure**
- Integrate with **electronic health record systems**
- Improve UI with **advanced medical visualization**

---

# 👨‍💻 Author

**Apoorv Raj**

Machine Learning & AI Engineer

---

⭐ If you found this project useful, consider giving it a **star on GitHub**.
