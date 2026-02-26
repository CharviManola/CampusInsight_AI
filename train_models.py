"""
CampusInsight AI - Model Training Script
Run: python train_models.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from preprocessing.preprocess import preprocess_all
from evaluation.evaluate import evaluate_classifier, evaluate_regressor, plot_feature_importance

os.makedirs('models', exist_ok=True)
os.makedirs('evaluation/plots', exist_ok=True)

def train_attendance_risk_model(df):
    print("\n[1] Training Attendance Risk Model (Logistic Regression)...")
    features = ['attendance', 'internal_marks', 'backlogs']
    X = df[features]
    y = (df['attendance'] < 75).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=500))
    ])
    pipeline.fit(X_train, y_train)

    metrics = evaluate_classifier(
        'Attendance Risk (LR)', pipeline, X_test, y_test,
        label_names=['Low Risk', 'High Risk']
    )
    joblib.dump(pipeline, 'models/attendance_risk_model.pkl')
    joblib.dump(features, 'models/attendance_features.pkl')
    print("✔ Saved: models/attendance_risk_model.pkl")
    return metrics

def train_dropout_model(df):
    print("\n[2] Training Dropout Prediction Model (Random Forest)...")
    features = ['attendance', 'cgpa', 'backlogs', 'financial_category_enc']
    X = df[features]
    y = df['dropout']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    metrics = evaluate_classifier(
        'Dropout Prediction (RF)', model, X_test, y_test,
        label_names=['Stay', 'Dropout']
    )
    plot_feature_importance('Dropout Prediction (RF)', model, features)
    joblib.dump(model, 'models/dropout_model.pkl')
    joblib.dump(features, 'models/dropout_features.pkl')
    print("✔ Saved: models/dropout_model.pkl")
    return metrics

def train_placement_model(df):
    print("\n[3] Training Placement Prediction Model (Gradient Boosting)...")
    features = ['cgpa', 'internships', 'communication_score', 'coding_score', 'backlogs']
    X = df[features]
    y = df['placed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                       max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_classifier(
        'Placement Prediction (GB)', model, X_test, y_test,
        label_names=['Not Placed', 'Placed']
    )
    plot_feature_importance('Placement Prediction (GB)', model, features)
    joblib.dump(model, 'models/placement_model.pkl')
    joblib.dump(features, 'models/placement_features.pkl')
    print("✔ Saved: models/placement_model.pkl")
    return metrics

def train_marks_regression_model(df):
    print("\n[4] Training Subject Performance Model (Linear Regression)...")
    features = ['attendance', 'internal_marks', 'cgpa']
    X = df[features]
    y = df['next_sem_marks']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    metrics = evaluate_regressor('Marks Prediction (LR)', pipeline, X_test, y_test)
    joblib.dump(pipeline, 'models/marks_model.pkl')
    joblib.dump(features, 'models/marks_features.pkl')
    print("✔ Saved: models/marks_model.pkl")
    return metrics

def plot_risk_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Risk Distribution Analysis', fontsize=16, fontweight='bold')

    axes[0].hist(df['attendance'], bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(75, color='red', linestyle='--', linewidth=2, label='75% Threshold')
    axes[0].set_title('Attendance Distribution')
    axes[0].set_xlabel('Attendance %')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    dropout_counts = df['dropout'].value_counts()
    axes[1].pie(dropout_counts, labels=['Retained', 'Dropped'],
                colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Dropout Distribution')

    plt.tight_layout()
    plt.savefig('evaluation/plots/risk_distribution.png', dpi=120)
    plt.close()
    print("✔ Saved: evaluation/plots/risk_distribution.png")

def plot_placement_bar(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    dept_data = df.groupby('financial_category')['placed'].mean().reset_index()
    colors = ['#6c5ce7', '#00b894', '#fd79a8', '#fdcb6e']
    ax.bar(dept_data['financial_category'], dept_data['placed'] * 100,
           color=colors[:len(dept_data)], edgecolor='white', linewidth=1.5)
    ax.set_title('Placement Rate by Financial Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Placement Rate (%)')
    ax.set_ylim(0, 100)
    for i, v in enumerate(dept_data['placed'] * 100):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('evaluation/plots/placement_bar.png', dpi=120)
    plt.close()
    print("✔ Saved: evaluation/plots/placement_bar.png")

def plot_marks_trend(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    df_sorted = df.sort_values('internal_marks').head(100)
    ax.plot(df_sorted['internal_marks'].values, label='Current Marks', color='#6c5ce7', linewidth=2)
    ax.plot(df_sorted['next_sem_marks'].values, label='Predicted Next Sem', color='#00b894',
            linewidth=2, linestyle='--')
    ax.fill_between(range(len(df_sorted)),
                    df_sorted['internal_marks'].values,
                    df_sorted['next_sem_marks'].values,
                    alpha=0.15, color='gray')
    ax.set_title('Marks Trend: Current vs Predicted Next Semester', fontsize=14, fontweight='bold')
    ax.set_xlabel('Student Index (sorted by current marks)')
    ax.set_ylabel('Marks')
    ax.legend()
    plt.tight_layout()
    plt.savefig('evaluation/plots/marks_trend.png', dpi=120)
    plt.close()
    print("✔ Saved: evaluation/plots/marks_trend.png")

if __name__ == '__main__':
    print("=" * 60)
    print("  CampusInsight AI - Training Pipeline")
    print("=" * 60)

    # Generate data if not exists
    if not os.path.exists('data/students.csv'):
        print("[DATA] Generating synthetic dataset...")
        exec(open('data/generate_data.py').read())

    df, le = preprocess_all()
    joblib.dump(le, 'models/label_encoder.pkl')
    print(f"[DATA] Loaded {len(df)} student records")
    print(f"[DATA] Columns: {list(df.columns)}")

    # Train all models
    att_metrics = train_attendance_risk_model(df)
    drop_metrics = train_dropout_model(df)
    place_metrics = train_placement_model(df)
    marks_metrics = train_marks_regression_model(df)

    # Visualizations
    plot_risk_distribution(df)
    plot_placement_bar(df)
    plot_marks_trend(df)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE - Summary")
    print("=" * 60)
    print(f"  Attendance Risk Accuracy : {att_metrics['accuracy']:.2%}")
    print(f"  Dropout Accuracy         : {drop_metrics['accuracy']:.2%}")
    print(f"  Placement Accuracy       : {place_metrics['accuracy']:.2%}")
    print(f"  Marks Prediction R²      : {marks_metrics['r2']:.4f}")
    print(f"  Marks Prediction RMSE    : {marks_metrics['rmse']:.4f}")
    print("=" * 60)
    print("\n✔ All models saved to /models/")
    print("✔ All plots saved to /evaluation/plots/")
    print("\nNext: Run  streamlit run app.py")
