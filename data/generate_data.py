import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

names = [f"Student_{i:03d}" for i in range(1, n+1)]
attendance = np.clip(np.random.normal(72, 15, n), 20, 100).round(1)
cgpa = np.clip(np.random.normal(6.8, 1.2, n), 4.0, 10.0).round(2)
backlogs = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.45, 0.25, 0.15, 0.08, 0.04, 0.03])
internships = np.random.choice([0, 1, 2, 3], n, p=[0.35, 0.40, 0.18, 0.07])
internal_marks = np.clip(np.random.normal(55, 15, n), 10, 80).round(1)
communication_score = np.clip(np.random.normal(60, 18, n), 10, 100).round(1)
coding_score = np.clip(np.random.normal(55, 20, n), 10, 100).round(1)
financial_category = np.random.choice(['General', 'OBC', 'SC', 'ST'], n, p=[0.40, 0.30, 0.20, 0.10])

# Dropout: high if low attendance + low cgpa + high backlogs
dropout_prob = (
    (100 - attendance) / 100 * 0.4 +
    (10 - cgpa) / 10 * 0.3 +
    backlogs / 5 * 0.3
)
dropout_prob += np.random.normal(0, 0.05, n)
dropout = (dropout_prob > 0.45).astype(int)

# Placed: high if high cgpa + internships + coding + low backlogs
placed_prob = (
    cgpa / 10 * 0.35 +
    internships / 3 * 0.25 +
    coding_score / 100 * 0.25 +
    communication_score / 100 * 0.15 -
    backlogs * 0.05
)
placed_prob += np.random.normal(0, 0.05, n)
placed = (placed_prob > 0.50).astype(int)

# Next semester marks
next_sem_marks = (
    attendance * 0.3 +
    internal_marks * 0.4 +
    cgpa * 3 +
    np.random.normal(0, 5, n)
).clip(20, 100).round(1)

df = pd.DataFrame({
    'id': range(1, n+1),
    'name': names,
    'attendance': attendance,
    'cgpa': cgpa,
    'backlogs': backlogs,
    'internships': internships,
    'internal_marks': internal_marks,
    'communication_score': communication_score,
    'coding_score': coding_score,
    'financial_category': financial_category,
    'dropout': dropout,
    'placed': placed,
    'next_sem_marks': next_sem_marks
})

df.to_csv('data/students.csv', index=False)
print(f"Dataset generated: {len(df)} rows")
print(df.describe())
