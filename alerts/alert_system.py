import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
EMAIL_USER = os.getenv('EMAIL_USER', '')
EMAIL_PASS = os.getenv('EMAIL_PASS', '')

def send_alert_email(to_email, student_name, risk_type, probability):
    if not EMAIL_USER or not EMAIL_PASS:
        print("[ALERT] Email credentials not configured. Skipping email.")
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = f"⚠️ CampusInsight AI Alert: {risk_type} Risk Detected"

        body = f"""
        <html><body>
        <h2 style="color:#d63031;">CampusInsight AI - Student Risk Alert</h2>
        <p>Dear Advisor,</p>
        <p>The following student has been flagged by our predictive model:</p>
        <table border="1" cellpadding="8" cellspacing="0">
            <tr><th>Student</th><td>{student_name}</td></tr>
            <tr><th>Risk Type</th><td>{risk_type}</td></tr>
            <tr><th>Probability</th><td>{probability:.1%}</td></tr>
        </table>
        <p>Please take immediate action and review the student's academic profile.</p>
        <p>— CampusInsight AI System</p>
        </body></html>
        """
        msg.attach(MIMEText(body, 'html'))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, to_email, msg.as_string())
        print(f"[ALERT] Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"[ALERT] Email failed: {e}")
        return False

def generate_risk_alerts(df, dropout_probs, placement_probs):
    alerts = []
    for i, row in df.iterrows():
        if dropout_probs[i] > 0.65:
            alerts.append({
                'student': row['name'],
                'risk': 'Dropout',
                'probability': dropout_probs[i],
                'severity': 'HIGH'
            })
        if row['attendance'] < 60:
            alerts.append({
                'student': row['name'],
                'risk': 'Low Attendance',
                'probability': 1 - row['attendance']/100,
                'severity': 'MEDIUM'
            })
        if placement_probs[i] < 0.25 and row.get('dropout', 0) == 0:
            alerts.append({
                'student': row['name'],
                'risk': 'Low Placement Chance',
                'probability': 1 - placement_probs[i],
                'severity': 'MEDIUM'
            })
    return alerts
