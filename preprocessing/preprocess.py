import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(path='data/students.csv'):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.copy()
    # Fill missing numerics with median
    num_cols = ['attendance', 'cgpa', 'backlogs', 'internships',
                'internal_marks', 'communication_score', 'coding_score']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    # Fill categorical
    df['financial_category'] = df['financial_category'].fillna('General')
    # Remove duplicates
    df = df.drop_duplicates(subset='id')
    return df

def encode_features(df):
    df = df.copy()
    le = LabelEncoder()
    df['financial_category_enc'] = le.fit_transform(df['financial_category'])
    return df, le

def feature_engineer(df):
    df = df.copy()
    df['risk_score'] = (
        (100 - df['attendance']) * 0.4 +
        df['backlogs'] * 10 +
        (80 - df['internal_marks']) * 0.3
    ).clip(0, 100)
    df['academic_strength'] = (df['cgpa'] * 10 + df['internal_marks']) / 2
    df['employability_index'] = (
        df['cgpa'] * 0.35 +
        df['coding_score'] / 100 * 10 * 0.35 +
        df['communication_score'] / 100 * 10 * 0.20 +
        df['internships'] * 1.5
    ).clip(0, 10)
    return df

def get_attendance_features(df):
    features = ['attendance', 'internal_marks', 'backlogs']
    X = df[features]
    y = (df['attendance'] < 75).astype(int)  # at-risk if <75%
    return X, y

def get_dropout_features(df):
    features = ['attendance', 'cgpa', 'backlogs', 'financial_category_enc']
    X = df[features]
    y = df['dropout']
    return X, y

def get_placement_features(df):
    features = ['cgpa', 'internships', 'communication_score', 'coding_score', 'backlogs']
    X = df[features]
    y = df['placed']
    return X, y

def get_marks_features(df):
    features = ['attendance', 'internal_marks', 'cgpa']
    X = df[features]
    y = df['next_sem_marks']
    return X, y

def preprocess_all(path='data/students.csv'):
    df = load_data(path)
    df = clean_data(df)
    df, le = encode_features(df)
    df = feature_engineer(df)
    return df, le
