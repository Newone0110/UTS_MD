"""
app_streamlit.py - Monolithic Streamlit App
=============================================
Jalankan dengan:
    streamlit run app_streamlit.py

Pastikan best_classification_model.pkl dan best_regression_model.pkl
berada di direktori yang sama dengan file ini.
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# CONFIG & SETUP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f77b4; }
    .sub-header  { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-card { background: #f0f4ff; padding: 1rem; border-radius: 10px;
                   border-left: 4px solid #1f77b4; margin-bottom: 0.5rem; }
    .placed-label     { color: #28a745; font-size: 1.5rem; font-weight: 700; }
    .not-placed-label { color: #dc3545; font-size: 1.5rem; font-weight: 700; }
    .salary-label     { color: #fd7e14; font-size: 1.5rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

@st.cache_resource
def load_models():
    with open("best_classification_model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("best_regression_model.pkl", "rb") as f:
        reg = pickle.load(f)
    return clf, reg

try:
    clf_model, reg_model = load_models()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─────────────────────────────────────────────
# SIDEBAR NAVIGASI
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
    st.markdown("### 🎓 Student Placement Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["🏠 Home", "🔮 Prediksi", "📊 Visualisasi Data", "ℹ️ Info Model"],
        index=0
    )
    st.markdown("---")
    st.markdown("**Dataset:** 5.000 mahasiswa")
    st.markdown("**Task:** Klasifikasi + Regresi")
    if model_loaded:
        st.success("✅ Model berhasil dimuat")
    else:
        st.error("❌ Model tidak ditemukan\nJalankan pipeline.py dulu")

# ─────────────────────────────────────────────
# HALAMAN HOME
# ─────────────────────────────────────────────

if page == "🏠 Home":
    st.markdown('<div class="main-header">🎓 Student Placement Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Prediksi status penempatan kerja dan estimasi gaji mahasiswa berbasis Machine Learning</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", "5.000 mahasiswa")
    with col2:
        st.metric("Fitur", "22 variabel")
    with col3:
        st.metric("Task", "Klasifikasi + Regresi")

    st.markdown("---")
    st.markdown("""
    ### 🗂️ Deskripsi Sistem
    Sistem ini menggunakan model Machine Learning yang telah dilatih untuk memprediksi:

    | Task | Target | Algoritma Terbaik |
    |------|--------|-------------------|
    | **Klasifikasi** | Status Penempatan (Placed / Not Placed) | Gradient Boosting / Random Forest |
    | **Regresi** | Estimasi Gaji (LPA) | Gradient Boosting / Random Forest |

    ### 📌 Cara Penggunaan
    1. Klik **🔮 Prediksi** di sidebar
    2. Isi data profil mahasiswa pada form
    3. Klik tombol **Prediksi** untuk mendapatkan hasil
    """)

# ─────────────────────────────────────────────
# HALAMAN PREDIKSI
# ─────────────────────────────────────────────

elif page == "🔮 Prediksi":
    st.markdown("## 🔮 Prediksi Placement & Salary")

    if not model_loaded:
        st.error("❌ Model tidak ditemukan. Jalankan `python pipeline.py` terlebih dahulu.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("### 📋 Data Akademik")
        col1, col2, col3 = st.columns(3)
        with col1:
            cgpa = st.slider("CGPA", 5.0, 10.0, 8.0, 0.01)
            tenth = st.slider("Nilai SMP (%)", 40.0, 100.0, 80.0, 0.1)
            twelfth = st.slider("Nilai SMA (%)", 40.0, 100.0, 80.0, 0.1)
        with col2:
            backlogs = st.number_input("Jumlah Backlogs", 0, 20, 0)
            attendance = st.slider("Attendance (%)", 50.0, 100.0, 85.0, 0.1)
            branch = st.selectbox("Jurusan", ["CSE", "ECE", "IT", "ME", "CE"])
        with col3:
            gender = st.selectbox("Gender", ["Male", "Female"])
            city_tier = st.selectbox("Tier Kota Asal", ["Tier 1", "Tier 2", "Tier 3"])
            family_income = st.selectbox("Pendapatan Keluarga", ["Low", "Medium", "High"])

        st.markdown("### 🛠️ Keterampilan Teknis")
        col4, col5, col6 = st.columns(3)
        with col4:
            coding = st.slider("Coding Skill (1-10)", 1, 10, 7)
            comm = st.slider("Communication Skill (1-10)", 1, 10, 7)
            aptitude = st.slider("Aptitude Skill (1-10)", 1, 10, 7)
        with col5:
            projects = st.number_input("Projects Selesai", 0, 20, 3)
            internships = st.number_input("Internships Selesai", 0, 10, 1)
            hackathons = st.number_input("Hackathon Diikuti", 0, 20, 2)
        with col6:
            certifications = st.number_input("Sertifikasi", 0, 20, 2)
            study_hours = st.slider("Jam Belajar/Hari", 0.0, 12.0, 4.0, 0.5)

        st.markdown("### 🧘 Faktor Gaya Hidup")
        col7, col8, col9 = st.columns(3)
        with col7:
            sleep_hours = st.slider("Jam Tidur/Hari", 4.0, 9.0, 7.0, 0.5)
            stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)
        with col8:
            part_time_job = st.selectbox("Kerja Part-Time?", ["No", "Yes"])
            internet_access = st.selectbox("Akses Internet?", ["Yes", "No"])
        with col9:
            extracurricular = st.selectbox("Kegiatan Ekstrakurikuler", ["Low", "Medium", "High"])

        submitted = st.form_submit_button("🚀 Prediksi Sekarang", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "gender": gender,
            "branch": branch,
            "cgpa": cgpa,
            "tenth_percentage": tenth,
            "twelfth_percentage": twelfth,
            "backlogs": backlogs,
            "study_hours_per_day": study_hours,
            "attendance_percentage": attendance,
            "projects_completed": projects,
            "internships_completed": internships,
            "coding_skill_rating": coding,
            "communication_skill_rating": comm,
            "aptitude_skill_rating": aptitude,
            "hackathons_participated": hackathons,
            "certifications_count": certifications,
            "sleep_hours": sleep_hours,
            "stress_level": stress_level,
            "part_time_job": part_time_job,
            "family_income_level": family_income,
            "city_tier": city_tier,
            "internet_access": internet_access,
            "extracurricular_involvement": extracurricular,
        }])

        clf_pred = clf_model.predict(input_data)[0]
        clf_prob = clf_model.predict_proba(input_data)[0]
        reg_pred = reg_model.predict(input_data)[0]

        st.markdown("---")
        st.markdown("## 📊 Hasil Prediksi")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### 🎯 Status Penempatan")
            if clf_pred == 1:
                st.markdown('<div class="placed-label">✅ PLACED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="not-placed-label">❌ NOT PLACED</div>', unsafe_allow_html=True)

            fig_prob = go.Figure(go.Bar(
                x=["Not Placed", "Placed"],
                y=[clf_prob[0]*100, clf_prob[1]*100],
                marker_color=["#dc3545", "#28a745"],
                text=[f"{clf_prob[0]*100:.1f}%", f"{clf_prob[1]*100:.1f}%"],
                textposition="auto"
            ))
            fig_prob.update_layout(title="Probabilitas Prediksi (%)", height=300,
                                   yaxis_title="Probabilitas (%)")
            st.plotly_chart(fig_prob, use_container_width=True)

        with col_r2:
            st.markdown("#### 💰 Estimasi Gaji")
            st.markdown(f'<div class="salary-label">Rp ~{reg_pred:.2f} LPA</div>', unsafe_allow_html=True)
            st.markdown(f"*≈ {reg_pred * 12:.0f}k/bulan dalam konversi relatif*")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=reg_pred,
                title={"text": "Estimasi Gaji (LPA)"},
                gauge={
                    "axis": {"range": [0, 20]},
                    "bar":  {"color": "#fd7e14"},
                    "steps": [
                        {"range": [0, 8],   "color": "#fff3cd"},
                        {"range": [8, 14],  "color": "#d4edda"},
                        {"range": [14, 20], "color": "#cce5ff"},
                    ],
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📋 Ringkasan Input")
        st.dataframe(input_data.T.rename(columns={0: "Nilai Input"}), use_container_width=True)

# ─────────────────────────────────────────────
# HALAMAN VISUALISASI DATA
# ─────────────────────────────────────────────

elif page == "📊 Visualisasi Data":
    st.markdown("## 📊 Eksplorasi & Visualisasi Dataset")

    try:
        df = pd.read_csv("A.csv")
        targets = pd.read_csv("A_targets.csv")
        df = df.merge(targets, on="Student_ID")

        tab1, tab2, tab3 = st.tabs(["📈 Distribusi Target", "🔗 Korelasi", "📦 Distribusi Fitur"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                counts = df["placement_status"].value_counts()
                fig = px.pie(values=counts.values, names=counts.index,
                             title="Distribusi Placement Status",
                             color_discrete_map={"Placed": "#28a745", "Not Placed": "#dc3545"})
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.histogram(df, x="salary_lpa", nbins=50,
                                    title="Distribusi Salary (LPA)",
                                    color_discrete_sequence=["#1f77b4"])
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            num_cols = ["cgpa", "tenth_percentage", "twelfth_percentage", "backlogs",
                        "coding_skill_rating", "communication_skill_rating",
                        "aptitude_skill_rating", "salary_lpa"]
            corr = df[num_cols].corr()
            fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                             title="Heatmap Korelasi Fitur Numerik",
                             color_continuous_scale="RdBu_r")
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            feat = st.selectbox("Pilih Fitur:", [
                "cgpa", "tenth_percentage", "twelfth_percentage", "backlogs",
                "study_hours_per_day", "attendance_percentage", "coding_skill_rating",
                "communication_skill_rating", "aptitude_skill_rating"
            ])
            fig4 = px.box(df, x="placement_status", y=feat, color="placement_status",
                          title=f"Distribusi {feat} berdasarkan Placement Status",
                          color_discrete_map={"Placed": "#28a745", "Not Placed": "#dc3545"})
            st.plotly_chart(fig4, use_container_width=True)

    except FileNotFoundError:
        st.warning("File A.csv atau A_targets.csv tidak ditemukan di direktori ini.")

# ─────────────────────────────────────────────
# HALAMAN INFO MODEL
# ─────────────────────────────────────────────

elif page == "ℹ️ Info Model":
    st.markdown("## ℹ️ Informasi Model")

    st.markdown("""
    ### 🏗️ Arsitektur Pipeline
    ```
    Input Data
        ↓
    ColumnTransformer
        ├── Numerik: Median Imputer → StandardScaler
        └── Kategorikal: Mode Imputer → OrdinalEncoder
        ↓
    Estimator (Classifier / Regressor)
        ↓
    Output (Prediction)
    ```

    ### 📊 Model yang Dibandingkan

    **Klasifikasi:**
    | Model | Keunggulan | Kelemahan |
    |-------|-----------|-----------|
    | Logistic Regression | Interpretable, cepat | Asumsi linearitas |
    | Random Forest | Robust terhadap outlier | Sulit diinterpretasi |
    | Gradient Boosting | Akurasi tinggi | Lebih lambat, risiko overfit |

    **Regresi:**
    | Model | Keunggulan | Kelemahan |
    |-------|-----------|-----------|
    | Ridge Regression | Menangani multikolinearitas | Asumsi linearitas |
    | Random Forest Regressor | Non-linear, robust | Memory intensive |
    | Gradient Boosting Regressor | Akurasi tinggi | Perlu tuning hyperparameter |

    ### 🛡️ Strategi Handling Data
    - **Missing Values**: Imputasi median (numerik), modus (kategorikal)
    - **Class Imbalance**: `class_weight='balanced'` pada model klasifikasi
    - **Salary=0 bias**: Regresi hanya menggunakan data mahasiswa Placed
    - **Scaling**: StandardScaler untuk fitur numerik
    """)
