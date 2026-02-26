"""
CampusInsight AI - Streamlit Dashboard
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CampusInsight AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fc; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
    }
    [data-testid="stSidebar"] * { color: white !important; }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #4a4e8b !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        margin: 15px 0 10px 0;
        font-weight: 700;
        font-size: 1.1rem;
    }

    /* Metric card style */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 2px 12px rgba(102,126,234,0.15);
        border-left: 4px solid #667eea;
        margin-bottom: 10px;
    }

    /* Risk badge */
    .risk-high {
        background-color: #ff4757;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-low {
        background-color: #2ed573;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-medium {
        background-color: #ffa502;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Divider */
    hr { border-color: #e0e0e0; }

    /* Alert box */
    .alert-box {
        background: #fff5f5;
        border: 1px solid #ff4757;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #c0392b;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Data & Models ──────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    path = 'data/students.csv'
    if not os.path.exists(path):
        os.system('python data/generate_data.py')
    return pd.read_csv(path)

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'attendance': 'models/attendance_risk_model.pkl',
        'dropout': 'models/dropout_model.pkl',
        'placement': 'models/placement_model.pkl',
        'marks': 'models/marks_model.pkl',
        'le': 'models/label_encoder.pkl',
    }
    for key, path in model_files.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models

df = load_dataset()
models = load_models()
models_loaded = all(k in models for k in ['attendance', 'dropout', 'placement', 'marks'])


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 CampusInsight AI")
    st.markdown("*Predictive Analytics for Smart Campus*")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📊 Data Overview",
        "📈 Model Performance",
        "🔮 Predict Student",
        "🚨 Risk Alerts",
        "📉 Trend Analysis"
    ])

    st.markdown("---")
    if not models_loaded:
        st.error("⚠️ Models not trained.\nRun: `python train_models.py`")
    else:
        st.success("✅ All models loaded")

    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.markdown(f"• Total Students: **{len(df)}**")
    st.markdown(f"• Features: **{len(df.columns)-2}**")
    st.markdown(f"• Dropout Rate: **{df['dropout'].mean():.1%}**")
    st.markdown(f"• Placement Rate: **{df['placed'].mean():.1%}**")


# ────────────────────────────────────────────────────────────────────────────
# PAGE 1: DATA OVERVIEW
# ────────────────────────────────────────────────────────────────────────────
if page == "📊 Data Overview":
    st.markdown("# 📊 Data Overview")
    st.markdown("Exploratory analysis of the student dataset used for model training.")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Students", len(df))
    col2.metric("Avg CGPA", f"{df['cgpa'].mean():.2f}")
    col3.metric("Avg Attendance", f"{df['attendance'].mean():.1f}%")
    col4.metric("Dropout Rate", f"{df['dropout'].mean():.1%}")
    col5.metric("Placement Rate", f"{df['placed'].mean():.1%}")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Attendance Distribution**")
        fig = px.histogram(df, x='attendance', nbins=30, color_discrete_sequence=['#667eea'],
                           labels={'attendance': 'Attendance %'})
        fig.add_vline(x=75, line_dash="dash", line_color="red",
                      annotation_text="75% Threshold", annotation_position="top")
        fig.update_layout(bargap=0.05, height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**CGPA Distribution**")
        fig2 = px.histogram(df, x='cgpa', nbins=30, color_discrete_sequence=['#764ba2'],
                            labels={'cgpa': 'CGPA'})
        fig2.update_layout(bargap=0.05, height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Backlogs Distribution**")
        backlog_counts = df['backlogs'].value_counts().sort_index()
        fig3 = px.bar(x=backlog_counts.index, y=backlog_counts.values,
                      labels={'x': 'Backlogs', 'y': 'Count'},
                      color_discrete_sequence=['#f093fb'])
        fig3.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown("**CGPA vs Placement**")
        fig4 = px.box(df, x='placed', y='cgpa', color='placed',
                      labels={'placed': 'Placed (0=No, 1=Yes)', 'cgpa': 'CGPA'},
                      color_discrete_map={0: '#ff4757', 1: '#2ed573'})
        fig4.update_layout(height=320, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("**Correlation Heatmap**")
    numeric_df = df[['attendance', 'cgpa', 'backlogs', 'internships',
                     'internal_marks', 'communication_score', 'coding_score',
                     'dropout', 'placed', 'next_sem_marks']]
    corr = numeric_df.corr()
    fig5 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, height=450)
    fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.markdown("**Raw Data Sample**")
    st.dataframe(df.head(20), use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 2: MODEL PERFORMANCE
# ────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown("# 📈 Model Performance Metrics")

    if not models_loaded:
        st.warning("⚠️ Train models first: `python train_models.py`")
        st.stop()

    st.markdown("### Classification Models")
    col1, col2, col3 = st.columns(3)

    # Quick test metrics using current data
    from preprocessing.preprocess import preprocess_all
    from sklearn.model_selection import cross_val_score

    df_proc, le = preprocess_all()

    # Attendance
    X_att = df_proc[['attendance', 'internal_marks', 'backlogs']]
    y_att = (df_proc['attendance'] < 75).astype(int)
    att_scores = cross_val_score(models['attendance'], X_att, y_att, cv=5, scoring='accuracy')
    col1.metric("Attendance Risk", f"{att_scores.mean():.2%}", "Logistic Regression")

    # Dropout
    X_drop = df_proc[['attendance', 'cgpa', 'backlogs', 'financial_category_enc']]
    y_drop = df_proc['dropout']
    drop_scores = cross_val_score(models['dropout'], X_drop, y_drop, cv=5, scoring='accuracy')
    col2.metric("Dropout Prediction", f"{drop_scores.mean():.2%}", "Random Forest")

    # Placement
    X_place = df_proc[['cgpa', 'internships', 'communication_score', 'coding_score', 'backlogs']]
    y_place = df_proc['placed']
    place_scores = cross_val_score(models['placement'], X_place, y_place, cv=5, scoring='accuracy')
    col3.metric("Placement Prediction", f"{place_scores.mean():.2%}", "Gradient Boosting")

    st.markdown("---")

    # Show saved plots
    plot_dir = 'evaluation/plots'
    if os.path.exists(plot_dir):
        plots = {
            'Dropout RF - Confusion Matrix': f'{plot_dir}/Dropout_Prediction_(RF)_cm.png',
            'Placement GB - Confusion Matrix': f'{plot_dir}/Placement_Prediction_(GB)_cm.png',
            'Dropout - Feature Importance': f'{plot_dir}/Dropout_Prediction_(RF)_importance.png',
            'Placement - Feature Importance': f'{plot_dir}/Placement_Prediction_(GB)_importance.png',
        }

        row1 = st.columns(2)
        row2 = st.columns(2)

        items = list(plots.items())
        for idx, (title, path) in enumerate(items):
            col = row1[idx] if idx < 2 else row2[idx - 2]
            with col:
                st.markdown(f"**{title}**")
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.info("Plot not found. Run `python train_models.py` first.")
    else:
        st.info("Run `python train_models.py` to generate evaluation plots.")

    # CV score comparison bar chart
    st.markdown("---")
    st.markdown("**Cross-Validation Accuracy Comparison (5-Fold)**")
    model_names = ['Attendance Risk\n(Logistic Reg)', 'Dropout\n(Random Forest)', 'Placement\n(Gradient Boost)']
    accs = [att_scores.mean(), drop_scores.mean(), place_scores.mean()]
    stds = [att_scores.std(), drop_scores.std(), place_scores.std()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_names,
        y=[a*100 for a in accs],
        error_y=dict(type='data', array=[s*100 for s in stds]),
        marker_color=['#667eea', '#764ba2', '#f093fb'],
        text=[f'{a:.1%}' for a in accs],
        textposition='outside'
    ))
    fig.update_layout(
        yaxis=dict(title='Accuracy (%)', range=[0, 110]),
        height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 3: PREDICT STUDENT
# ────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict Student":
    st.markdown("# 🔮 Student Prediction Panel")
    st.markdown("Enter student details to get ML-powered risk and performance predictions.")

    if not models_loaded:
        st.warning("⚠️ Train models first: `python train_models.py`")
        st.stop()

    with st.form("student_form"):
        st.markdown("### 📋 Student Academic Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Student Name", value="Student_X")
            attendance = st.slider("Attendance (%)", 20.0, 100.0, 72.0, 0.5)
            cgpa = st.slider("CGPA", 4.0, 10.0, 7.0, 0.1)
        with col2:
            internal_marks = st.slider("Internal Marks (/80)", 10.0, 80.0, 55.0, 0.5)
            backlogs = st.number_input("Backlogs", 0, 10, 0)
            internships = st.number_input("Internships Completed", 0, 5, 0)
        with col3:
            communication_score = st.slider("Communication Score (/100)", 10.0, 100.0, 60.0, 0.5)
            coding_score = st.slider("Coding Score (/100)", 10.0, 100.0, 55.0, 0.5)
            financial_category = st.selectbox("Financial Category", ['General', 'OBC', 'SC', 'ST'])

        submitted = st.form_submit_button("🚀 Run Predictions", use_container_width=True)

    if submitted:
        st.markdown("---")
        st.markdown("## 🔍 Prediction Results")

        le = models.get('le')
        fin_enc = le.transform([financial_category])[0] if le else 0

        # ── 1. Attendance Risk
        att_input = np.array([[attendance, internal_marks, backlogs]])
        att_prob = models['attendance'].predict_proba(att_input)[0][1]
        att_risk = "HIGH RISK" if att_prob > 0.5 else "LOW RISK"
        att_color = "risk-high" if att_prob > 0.5 else "risk-low"

        # ── 2. Dropout
        drop_input = np.array([[attendance, cgpa, backlogs, fin_enc]])
        drop_prob = models['dropout'].predict_proba(drop_input)[0][1]
        drop_risk = "HIGH" if drop_prob > 0.65 else "MEDIUM" if drop_prob > 0.4 else "LOW"
        drop_color = "risk-high" if drop_prob > 0.65 else "risk-medium" if drop_prob > 0.4 else "risk-low"

        # ── 3. Placement
        place_input = np.array([[cgpa, internships, communication_score, coding_score, backlogs]])
        place_prob = models['placement'].predict_proba(place_input)[0][1]
        place_status = "LIKELY" if place_prob > 0.5 else "UNLIKELY"
        place_color = "risk-low" if place_prob > 0.5 else "risk-high"

        # ── 4. Next Sem Marks
        marks_input = np.array([[attendance, internal_marks, cgpa]])
        predicted_marks = models['marks'].predict(marks_input)[0]

        # Display
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;color:#666;">Attendance Risk</div>
                <div style="font-size:1.8rem;font-weight:700;color:#4a4e8b;">{att_prob:.1%}</div>
                <span class="{att_color}">{att_risk}</span>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;color:#666;">Dropout Probability</div>
                <div style="font-size:1.8rem;font-weight:700;color:#4a4e8b;">{drop_prob:.1%}</div>
                <span class="{drop_color}">{drop_risk} RISK</span>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;color:#666;">Placement Probability</div>
                <div style="font-size:1.8rem;font-weight:700;color:#4a4e8b;">{place_prob:.1%}</div>
                <span class="{place_color}">{place_status}</span>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;color:#666;">Predicted Next Sem Marks</div>
                <div style="font-size:1.8rem;font-weight:700;color:#4a4e8b;">{predicted_marks:.1f}</div>
                <span class="risk-low">REGRESSION ESTIMATE</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Probability gauge chart
        st.markdown("### 📊 Probability Score Dashboard")
        fig = go.Figure()

        metrics_data = {
            'Attendance Risk': att_prob,
            'Dropout Risk': drop_prob,
            'Placement Chance': place_prob,
        }
        colors_bar = ['#ff4757' if v > 0.6 else '#ffa502' if v > 0.35 else '#2ed573'
                      for v in metrics_data.values()]

        fig.add_trace(go.Bar(
            x=list(metrics_data.keys()),
            y=[v * 100 for v in metrics_data.values()],
            marker_color=colors_bar,
            text=[f'{v:.1%}' for v in metrics_data.values()],
            textposition='outside',
            width=0.4
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            yaxis=dict(title='Probability (%)', range=[0, 115]),
            height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk alerts
        st.markdown("### 🚨 Risk Alerts")
        alerts_found = False
        if att_prob > 0.5:
            st.markdown(f'<div class="alert-box">⚠️ <b>Attendance Alert:</b> {name} is at HIGH risk due to low attendance. Probability: {att_prob:.1%}</div>', unsafe_allow_html=True)
            alerts_found = True
        if drop_prob > 0.5:
            st.markdown(f'<div class="alert-box">🚨 <b>Dropout Alert:</b> {name} has {drop_prob:.1%} dropout probability. Immediate counseling recommended.</div>', unsafe_allow_html=True)
            alerts_found = True
        if place_prob < 0.3:
            st.markdown(f'<div class="alert-box">📉 <b>Placement Alert:</b> {name} has low placement probability ({place_prob:.1%}). Recommend skill development programs.</div>', unsafe_allow_html=True)
            alerts_found = True
        if not alerts_found:
            st.success(f"✅ {name} shows no critical risk indicators. Keep up the good work!")


# ────────────────────────────────────────────────────────────────────────────
# PAGE 4: RISK ALERTS
# ────────────────────────────────────────────────────────────────────────────
elif page == "🚨 Risk Alerts":
    st.markdown("# 🚨 Batch Risk Alert Analysis")
    st.markdown("Automated risk classification for all students in the dataset.")

    if not models_loaded:
        st.warning("⚠️ Train models first: `python train_models.py`")
        st.stop()

    from preprocessing.preprocess import preprocess_all
    df_proc, le = preprocess_all()

    X_drop = df_proc[['attendance', 'cgpa', 'backlogs', 'financial_category_enc']].values
    X_place = df_proc[['cgpa', 'internships', 'communication_score', 'coding_score', 'backlogs']].values

    drop_probs = models['dropout'].predict_proba(X_drop)[:, 1]
    place_probs = models['placement'].predict_proba(X_place)[:, 1]

    df_risk = df_proc[['name', 'attendance', 'cgpa', 'backlogs', 'internal_marks']].copy()
    df_risk['Dropout Prob'] = (drop_probs * 100).round(1)
    df_risk['Placement Prob'] = (place_probs * 100).round(1)
    df_risk['Dropout Risk'] = pd.cut(drop_probs, bins=[0, 0.35, 0.65, 1.0],
                                     labels=['Low', 'Medium', 'High'])
    df_risk['Low Attendance'] = df_proc['attendance'] < 75

    col1, col2, col3 = st.columns(3)
    col1.metric("High Dropout Risk", int((drop_probs > 0.65).sum()),
                f"{(drop_probs > 0.65).mean():.1%} of students")
    col2.metric("Low Attendance", int((df_proc['attendance'] < 75).sum()),
                f"{(df_proc['attendance'] < 75).mean():.1%} of students")
    col3.metric("Low Placement Chance", int((place_probs < 0.25).sum()),
                f"{(place_probs < 0.25).mean():.1%} of students")

    st.markdown("---")
    st.markdown("### 🔴 High Risk Students (Dropout Probability > 65%)")
    high_risk = df_risk[drop_probs > 0.65].sort_values('Dropout Prob', ascending=False)
    if not high_risk.empty:
        st.dataframe(high_risk.head(30), use_container_width=True)
    else:
        st.success("No high-risk students detected.")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Dropout Risk Distribution**")
        risk_counts = df_risk['Dropout Risk'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                     color_discrete_map={'Low': '#2ed573', 'Medium': '#ffa502', 'High': '#ff4757'},
                     hole=0.4)
        fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Dropout Probability Histogram**")
        fig2 = px.histogram(x=drop_probs * 100, nbins=30, color_discrete_sequence=['#ff4757'],
                            labels={'x': 'Dropout Probability (%)'})
        fig2.add_vline(x=65, line_dash="dash", line_color="darkred",
                       annotation_text="High Risk Threshold")
        fig2.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# PAGE 5: TREND ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
elif page == "📉 Trend Analysis":
    st.markdown("# 📉 Trend & Performance Analysis")

    plot_dir = 'evaluation/plots'

    tab1, tab2, tab3 = st.tabs(["📈 Marks Trend", "📊 Placement Analysis", "⚠️ Risk Distribution"])

    with tab1:
        marks_plot = f'{plot_dir}/marks_trend.png'
        if os.path.exists(marks_plot):
            st.image(marks_plot, use_container_width=True)
        else:
            st.info("Run `python train_models.py` to generate trend plots.")

        st.markdown("---")
        st.markdown("**Attendance vs Next Sem Marks (Interactive)**")
        fig = px.scatter(df, x='attendance', y='next_sem_marks', color='dropout',
                         color_discrete_map={0: '#2ed573', 1: '#ff4757'},
                         hover_data=['name', 'cgpa', 'backlogs'],
                         labels={'attendance': 'Attendance %', 'next_sem_marks': 'Next Sem Marks',
                                 'dropout': 'Dropout'},
                         opacity=0.65)
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        placement_plot = f'{plot_dir}/placement_bar.png'
        if os.path.exists(placement_plot):
            st.image(placement_plot, use_container_width=True)
        else:
            st.info("Run `python train_models.py` first.")

        st.markdown("---")
        st.markdown("**CGPA vs Placement Probability**")
        if models_loaded:
            cgpa_range = np.linspace(4, 10, 60)
            probs = []
            for c in cgpa_range:
                inp = np.array([[c, 1, 60, 55, 0]])
                probs.append(models['placement'].predict_proba(inp)[0][1])
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=cgpa_range, y=[p*100 for p in probs],
                mode='lines', line=dict(color='#667eea', width=3),
                fill='tozeroy', fillcolor='rgba(102,126,234,0.15)'
            ))
            fig3.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5)
            fig3.update_layout(
                xaxis_title='CGPA', yaxis_title='Placement Probability (%)',
                height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        risk_plot = f'{plot_dir}/risk_distribution.png'
        if os.path.exists(risk_plot):
            st.image(risk_plot, use_container_width=True)
        else:
            st.info("Run `python train_models.py` first.")

        st.markdown("---")
        st.markdown("**Backlogs vs Dropout Rate**")
        backlog_dropout = df.groupby('backlogs')['dropout'].mean().reset_index()
        fig4 = px.bar(backlog_dropout, x='backlogs', y='dropout',
                      color='dropout',
                      color_continuous_scale='RdYlGn_r',
                      labels={'dropout': 'Dropout Rate', 'backlogs': 'Number of Backlogs'})
        fig4.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)
