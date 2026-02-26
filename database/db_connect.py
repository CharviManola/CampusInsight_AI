import mysql.connector
import pandas as pd
import os

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'campusinsight')
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS students (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    attendance FLOAT,
    cgpa FLOAT,
    backlogs INT,
    internships INT,
    internal_marks FLOAT,
    communication_score FLOAT,
    coding_score FLOAT,
    financial_category VARCHAR(20),
    dropout INT,
    placed INT,
    next_sem_marks FLOAT
);
"""

def get_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"[DB] Connection failed: {e}")
        return None

def setup_database():
    conn = get_connection()
    if not conn:
        return False
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    cursor.execute(f"USE {DB_CONFIG['database']}")
    cursor.execute(CREATE_TABLE_SQL)
    conn.commit()
    cursor.close()
    conn.close()
    print("[DB] Database and table created successfully.")
    return True

def insert_students_from_csv(csv_path='data/students.csv'):
    conn = get_connection()
    if not conn:
        return
    df = pd.read_csv(csv_path)
    cursor = conn.cursor()
    sql = """
    INSERT IGNORE INTO students
    (id, name, attendance, cgpa, backlogs, internships, internal_marks,
     communication_score, coding_score, financial_category, dropout, placed, next_sem_marks)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    for _, row in df.iterrows():
        cursor.execute(sql, tuple(row))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[DB] Inserted {len(df)} records.")

def fetch_all_students():
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

if __name__ == '__main__':
    setup_database()
    insert_students_from_csv()
