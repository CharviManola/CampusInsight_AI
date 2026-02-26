₹# 🎓 CampusInsight AI
### A Predictive Data Science Framework for Smart Campus Analytics

> Final Year Major Project | Data Science & Machine Learning

---

## 📌 Project Overview

CampusInsight AI is a complete machine learning analytics platform for predicting student academic outcomes. It leverages supervised learning models to identify at-risk students, predict placement success, estimate academic performance, and trigger automated alerts — all visualized in an analytical Streamlit dashboard.

---

## 🧠 Machine Learning Models

| # | Model | Algorithm | Task |
|---|-------|-----------|------|
| 1 | Attendance Risk Predictor | Logistic Regression | Binary Classification |
| 2 | Dropout Predictor | Random Forest Classifier | Binary Classification |
| 3 | Placement Predictor | Gradient Boosting Classifier | Binary Classification |
| 4 | Marks Performance | Linear Regression | Regression |

---

## 🗂️ Project Structure

```
CampusInsightAI/
│
├── data/
│   ├── generate_data.py       # Synthetic dataset generator (500 students)
│   └── students.csv           # Generated after running generate_data.py
│
├── preprocessing/
│   └── preprocess.py          # Data cleaning, encoding, feature engineering
│
├── models/                    # Saved .pkl model files (after training)
│   ├── attendance_risk_model.pkl
│   ├── dropout_model.pkl
│   ├── placement_model.pkl
│   ├── marks_model.pkl
│   └── label_encoder.pkl
│
├── evaluation/
│   ├── evaluate.py            # Metrics: accuracy, confusion matrix, R², RMSE
│   └── plots/                 # Auto-generated evaluation plots
│
├── dashboard/                 # (App logic in app.py)
│
├── database/
│   └── db_connect.py          # MySQL connection and data insertion
│
├── alerts/
│   └── alert_system.py        # SMTP email alerts for high-risk students
│
├── train_models.py            # 🚀 Main training script
├── app.py                     # 🎯 Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download
```bash
cd CampusInsightAI
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Dataset
```bash
python data/generate_data.py
```

### 5. Train Models
```bash
python train_models.py
```
This will:
- Preprocess the data
- Train all 4 ML models
- Save `.pkl` model files to `/models/`
- Generate evaluation plots in `/evaluation/plots/`
- Print accuracy, AUC, RMSE, R² scores

### 6. Launch Dashboard
```bash
streamlit run app.py
```
Open browser at: `http://localhost:8501`

---

## 🗄️ MySQL Database Setup (Optional)

If you want to use MySQL for data storage:

1. Create a MySQL user and note credentials
2. Set environment variables:
```bash
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=yourpassword
export DB_NAME=campusinsight
```
3. Run the database setup:
```bash
python database/db_connect.py
```

---

## 📧 Email Alerts Setup (Optional)

Set the following environment variables for SMTP alerts:
```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export EMAIL_USER=youremail@gmail.com
export EMAIL_PASS=yourapppassword
```

---

## 📊 Dashboard Sections

| Section | Description |
|---------|-------------|
| 📊 Data Overview | EDA, distributions, correlation heatmap |
| 📈 Model Performance | CV accuracy, confusion matrices, feature importance |
| 🔮 Predict Student | Input form → real-time probability predictions |
| 🚨 Risk Alerts | Batch analysis, high-risk student table |
| 📉 Trend Analysis | Marks trend, CGPA vs placement curve, risk histograms |

---

## 🔬 ML Workflow Implemented

- ✅ Data generation (500 synthetic realistic records)
- ✅ Data cleaning & missing value handling
- ✅ Feature engineering (risk score, academic strength, employability index)
- ✅ Label encoding for categorical features
- ✅ Train-test split (80/20)
- ✅ 5-fold cross-validation
- ✅ `predict_proba()` for probability scoring
- ✅ Confusion matrix + classification report
- ✅ RMSE + R² for regression
- ✅ Feature importance visualization
- ✅ Model saving with `joblib`
- ✅ Interactive Streamlit dashboard

---

## 📦 Tech Stack

- **Language**: Python 3.9+
- **ML**: Scikit-learn (LogisticRegression, RandomForest, GradientBoosting, LinearRegression)
- **Data**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Storage**: MySQL (optional), CSV
- **Model Persistence**: Joblib
- **Alerts**: SMTP (optional)

---

## 👨‍💻 Author

**CampusInsight AI** — Final Year Data Science Project  
Built with Python | Scikit-learn | Streamlit
