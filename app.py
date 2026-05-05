# -*- coding: utf-8 -*-
"""
ALDIMI-PREDICT | Dashboard de Clasificación de Riesgo de Salud
Motor: MLPClassifier sobre global_cancer_patients_2015_2024
Ejecutar: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier

# ──────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ALDIMI-PREDICT | Riesgo Oncológico",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# ESTILOS CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1f3d 0%, #1a3560 100%);
}
[data-testid="stSidebar"] * { color: #e8edf5 !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label { color: #a8bdda !important; font-size: 0.82rem !important; }

/* Métricas */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e8ecf2;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #6b7a99 !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; }

/* Tarjeta resultado */
.result-card {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}
.result-card.alto  { background: linear-gradient(135deg, #fee2e2, #fecaca); border-left: 6px solid #ef4444; }
.result-card.medio { background: linear-gradient(135deg, #fef9c3, #fde68a); border-left: 6px solid #f59e0b; }
.result-card.bajo  { background: linear-gradient(135deg, #dcfce7, #bbf7d0); border-left: 6px solid #22c55e; }
.result-card h1 { font-family: 'DM Serif Display', serif; font-size: 2.4rem; margin: 0; }
.result-card p  { font-size: 0.95rem; margin: 6px 0 0; opacity: 0.8; }

/* Header */
.page-header {
    background: linear-gradient(135deg, #0f1f3d 0%, #1e3a7a 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.page-header h1 { color: white; font-family: 'DM Serif Display', serif; font-size: 2rem; margin: 0; }
.page-header p  { color: #a8bdda; margin: 4px 0 0; font-size: 0.9rem; }

/* Tab section */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e3a7a;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e8ecf2;
}

/* Alerta crítica */
.alert-box {
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.alert-alto  { background: #fff1f2; border-left: 4px solid #ef4444; color: #7f1d1d; }
.alert-medio { background: #fffbeb; border-left: 4px solid #f59e0b; color: #78350f; }
.alert-bajo  { background: #f0fdf4; border-left: 4px solid #22c55e; color: #14532d; }

/* Botón principal */
.stButton > button {
    background: linear-gradient(135deg, #1e3a7a, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(37,99,235,0.4) !important;
}

/* Tabla historial */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Probabilidades */
.prob-bar-container { margin: 8px 0; }
.prob-label { font-size: 0.82rem; color: #4b5563; margin-bottom: 3px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CARGA Y ENTRENAMIENTO DEL MODELO
# ──────────────────────────────────────────────
CANCER_TYPES  = ["Breast", "Colon", "Leukemia", "Liver", "Lung", "Prostate", "Skin", "Cervical"]
CANCER_STAGES = ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"]
COUNTRIES     = ["Australia", "Brazil", "Canada", "China", "Germany", "India", "Pakistan", "Russia", "UK", "USA"]
GENDERS       = ["Male", "Female", "Other"]

STAGE_MAP = {"Stage 0": 1, "Stage I": 2, "Stage II": 3, "Stage III": 4, "Stage IV": 5}

@st.cache_resource(show_spinner="Entrenando modelos...")
def load_and_train():
    """Carga el dataset desde Kaggle y entrena los modelos."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("zahidmughal2343/global-cancer-patients-2015-2024")
        path = os.path.join(path, "global_cancer_patients_2015_2024.csv")
        df = pd.read_csv(path)
        dataset_loaded = True
    except Exception:
        # Demo con datos sintéticos si no hay Kaggle disponible
        np.random.seed(42)
        n = 50000
        df = pd.DataFrame({
            "Patient_ID": [f"P{i:05d}" for i in range(n)],
            "Age": np.random.randint(20, 91, n),
            "Gender": np.random.choice(GENDERS, n),
            "Country_Region": np.random.choice(COUNTRIES, n),
            "Year": np.random.randint(2015, 2025, n),
            "Genetic_Risk": np.random.uniform(0, 10, n),
            "Air_Pollution": np.random.uniform(0, 10, n),
            "Alcohol_Use": np.random.uniform(0, 10, n),
            "Smoking": np.random.uniform(0, 10, n),
            "Obesity_Level": np.random.uniform(0, 10, n),
            "Cancer_Type": np.random.choice(CANCER_TYPES, n),
            "Cancer_Stage": np.random.choice(CANCER_STAGES, n),
            "Treatment_Cost_USD": np.random.uniform(5000, 100000, n),
            "Survival_Years": np.random.uniform(0, 10, n),
            "Target_Severity_Score": np.random.normal(5, 1.2, n).clip(0.9, 9.16),
        })
        dataset_loaded = False

    # Preprocesamiento
    df_proc = df.copy()
    df_proc = df_proc.drop(columns=["Patient_ID"])
    df_proc["Cancer_Stage"] = df_proc["Cancer_Stage"].map(STAGE_MAP)
    df_proc = pd.get_dummies(df_proc, drop_first=True)
    df_proc["Severity_Class"] = pd.cut(
        df_proc["Target_Severity_Score"],
        bins=[0, 3, 7, 10], labels=[0, 1, 2]
    )

    X = df_proc.drop(columns=["Target_Severity_Score", "Severity_Class"])
    y = df_proc["Severity_Class"]
    feature_cols = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(5, 3, 7, 2), max_iter=1000, random_state=1)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)

    # Baseline: Árbol de decisión
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    return {
        "mlp": mlp, "dt": dt, "scaler": scaler,
        "X_test": X_test, "y_test": y_test,
        "y_pred_mlp": y_pred_mlp, "y_pred_dt": y_pred_dt,
        "feature_cols": feature_cols,
        "dataset_loaded": dataset_loaded,
        "n_train": len(X_train), "n_test": len(X_test),
        "df_raw": df,
    }


def build_patient_vector(patient_data: dict, feature_cols: list) -> np.ndarray:
    """Construye el vector de features para un paciente nuevo."""
    row = {col: 0 for col in feature_cols}

    # Numéricas directas
    for k in ["Age", "Year", "Genetic_Risk", "Air_Pollution",
              "Alcohol_Use", "Smoking", "Obesity_Level",
              "Treatment_Cost_USD", "Survival_Years"]:
        if k in row:
            row[k] = patient_data.get(k, 0)

    # Cancer_Stage ordinal
    if "Cancer_Stage" in row:
        row["Cancer_Stage"] = STAGE_MAP.get(patient_data.get("Cancer_Stage", "Stage 0"), 1)

    # get_dummies columns (drop_first=True → primera categoría = 0)
    gender = patient_data.get("Gender", "Female")
    if f"Gender_{gender}" in row:
        row[f"Gender_{gender}"] = 1

    country = patient_data.get("Country_Region", "Australia")
    if f"Country_Region_{country}" in row:
        row[f"Country_Region_{country}"] = 1

    cancer_type = patient_data.get("Cancer_Type", "Breast")
    if f"Cancer_Type_{cancer_type}" in row:
        row[f"Cancer_Type_{cancer_type}"] = 1

    return np.array([list(row.values())])


def get_priority_info(cls):
    labels = {0: ("BAJO", "bajo", "🟢", "#22c55e"),
              1: ("MEDIO", "medio", "🟡", "#f59e0b"),
              2: ("ALTO", "alto", "🔴", "#ef4444")}
    return labels.get(int(cls), ("—", "bajo", "⚪", "#999"))


# ──────────────────────────────────────────────
# CARGA
# ──────────────────────────────────────────────
data = load_and_train()
mlp       = data["mlp"]
dt        = data["dt"]
scaler    = data["scaler"]
X_test    = data["X_test"]
y_test    = data["y_test"]
y_pred_mlp = data["y_pred_mlp"]
y_pred_dt  = data["y_pred_dt"]
feature_cols = data["feature_cols"]
df_raw    = data["df_raw"]

if "historial" not in st.session_state:
    st.session_state.historial = []


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ALDIMI-PREDICT")
    st.markdown("*Motor de Clasificación de Riesgo Oncológico*")
    st.markdown("---")
    st.markdown("### 📋 Datos del Paciente")

    age    = st.slider("Edad", 20, 90, 50)
    gender = st.selectbox("Género", GENDERS)
    country = st.selectbox("País / Región", COUNTRIES)
    year   = st.selectbox("Año de diagnóstico", list(range(2015, 2025)), index=9)

    st.markdown("#### 🧬 Factores de Riesgo (0–10)")
    gen_risk = st.slider("Riesgo Genético",   0.0, 10.0, 5.0, 0.1)
    air_poll = st.slider("Contaminación Aire", 0.0, 10.0, 5.0, 0.1)
    alcohol  = st.slider("Consumo de Alcohol", 0.0, 10.0, 5.0, 0.1)
    smoking  = st.slider("Tabaquismo",          0.0, 10.0, 5.0, 0.1)
    obesity  = st.slider("Nivel de Obesidad",   0.0, 10.0, 5.0, 0.1)

    st.markdown("#### 🩺 Datos Clínicos")
    cancer_type  = st.selectbox("Tipo de Cáncer", CANCER_TYPES)
    cancer_stage = st.selectbox("Etapa del Cáncer", CANCER_STAGES)
    cost         = st.number_input("Costo Tratamiento (USD)", 5000, 100000, 50000, step=1000)
    survival     = st.slider("Años de Supervivencia", 0.0, 10.0, 5.0, 0.1)

    st.markdown("---")
    clasificar_btn = st.button("🔍 Clasificar Paciente")

    if not data["dataset_loaded"]:
        st.warning("⚠️ Kaggle no disponible. Usando datos sintéticos de demostración.")


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div style="font-size:2.8rem;">🩺</div>
    <div>
        <h1>ALDIMI-PREDICT</h1>
        <p>Motor de Clasificación de Riesgo de Salud Oncológico · Machine Learning 1ACC0057</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TABS PRINCIPALES
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Clasificación Individual",
    "📊 Métricas del Modelo",
    "📈 Comparativa de Algoritmos",
    "🗂️ Historial de Pacientes"
])


# ══════════════════════════════════════════════
# TAB 1 — CLASIFICACIÓN INDIVIDUAL
# ══════════════════════════════════════════════
with tab1:
    col_res, col_info = st.columns([1, 1], gap="large")

    with col_res:
        st.markdown('<div class="section-title">Resultado de Clasificación</div>', unsafe_allow_html=True)

        if clasificar_btn:
            patient_data = {
                "Age": age, "Gender": gender, "Country_Region": country,
                "Year": year, "Genetic_Risk": gen_risk, "Air_Pollution": air_poll,
                "Alcohol_Use": alcohol, "Smoking": smoking, "Obesity_Level": obesity,
                "Cancer_Type": cancer_type, "Cancer_Stage": cancer_stage,
                "Treatment_Cost_USD": cost, "Survival_Years": survival,
            }

            vec = build_patient_vector(patient_data, feature_cols)
            vec_scaled = scaler.transform(vec)

            pred_class  = int(mlp.predict(vec_scaled)[0])
            pred_proba  = mlp.predict_proba(vec_scaled)[0]
            label, css, icon, color = get_priority_info(pred_class)

            # Tarjeta resultado
            descs = {
                0: "Paciente con baja urgencia. Monitoreo rutinario recomendado.",
                1: "Paciente que requiere seguimiento activo y evaluación periódica.",
                2: "⚠️ Paciente crítico. Requiere intervención inmediata y prioritaria."
            }
            st.markdown(f"""
            <div class="result-card {css}">
                <div style="font-size:3.5rem;">{icon}</div>
                <h1>RIESGO {label}</h1>
                <p>{descs[pred_class]}</p>
            </div>
            """, unsafe_allow_html=True)

            # Probabilidades
            st.markdown("**Distribución de probabilidades:**")
            prob_labels = {0: ("Bajo", "#22c55e"), 1: ("Medio", "#f59e0b"), 2: ("Alto", "#ef4444")}
            for i, (pl, pc) in prob_labels.items():
                pval = pred_proba[i] * 100
                st.markdown(f'<div class="prob-label">{pl}: {pval:.1f}%</div>', unsafe_allow_html=True)
                st.progress(float(pred_proba[i]))

            # Guardar historial
            st.session_state.historial.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Edad": age, "Género": gender, "País": country,
                "Tipo Cáncer": cancer_type, "Etapa": cancer_stage,
                "Prioridad": label, "Confianza (%)": f"{max(pred_proba)*100:.1f}%"
            })

            if pred_class == 2:
                st.markdown('<div class="alert-box alert-alto">🚨 Este paciente ha sido marcado como ALTA prioridad. Notificar al equipo médico de inmediato.</div>', unsafe_allow_html=True)
            elif pred_class == 1:
                st.markdown('<div class="alert-box alert-medio">⏰ Programar evaluación médica en los próximos 7 días.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-box alert-bajo">✅ Continuar con el protocolo de monitoreo estándar.</div>', unsafe_allow_html=True)

        else:
            st.info("👈 Completa los datos del paciente en el panel izquierdo y presiona **Clasificar Paciente**.")

    with col_info:
        st.markdown('<div class="section-title">Resumen de Datos Ingresados</div>', unsafe_allow_html=True)
        resumen = {
            "Campo": ["Edad", "Género", "País", "Año Diagnóstico",
                      "Riesgo Genético", "Contaminación", "Alcohol", "Tabaquismo", "Obesidad",
                      "Tipo de Cáncer", "Etapa", "Costo Tratamiento", "Años Supervivencia"],
            "Valor": [age, gender, country, year,
                      f"{gen_risk:.1f}/10", f"{air_poll:.1f}/10", f"{alcohol:.1f}/10",
                      f"{smoking:.1f}/10", f"{obesity:.1f}/10",
                      cancer_type, cancer_stage, f"${cost:,}", f"{survival:.1f} años"]
        }
        st.dataframe(pd.DataFrame(resumen), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Escala de Prioridad</div>', unsafe_allow_html=True)
        escala = pd.DataFrame({
            "Clase": [0, 1, 2],
            "Prioridad": ["🟢 Bajo", "🟡 Medio", "🔴 Alto"],
            "Rango Score": ["[0.0, 3.0]", "(3.0, 7.0]", "(7.0, 10.0]"],
            "Acción": ["Monitoreo rutinario", "Evaluación periódica", "Intervención inmediata"]
        })
        st.dataframe(escala, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 2 — MÉTRICAS DEL MODELO
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Métricas de Desempeño — MLPClassifier</div>', unsafe_allow_html=True)

    # KPIs
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    acc   = accuracy_score(y_test, y_pred_mlp)
    f1m   = f1_score(y_test, y_pred_mlp, average="macro", zero_division=0)
    rec_h = recall_score(y_test, y_pred_mlp, labels=[2], average="macro", zero_division=0)
    pre_h = precision_score(y_test, y_pred_mlp, labels=[2], average="macro", zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy Global", f"{acc:.3f}", f"{'✅' if acc >= 0.80 else '⚠️'} {'OK' if acc >= 0.80 else 'Mejorar'}")
    c2.metric("Macro F1-Score",  f"{f1m:.3f}",  f"{'✅' if f1m >= 0.75 else '⚠️'}")
    c3.metric("Recall — Alto",   f"{rec_h:.3f}", f"{'✅' if rec_h >= 0.85 else '⚠️'} Crítico")
    c4.metric("Precisión — Alto",f"{pre_h:.3f}", f"{'✅' if pre_h >= 0.80 else '⚠️'}")

    st.markdown("---")

    col_cm, col_cr = st.columns([1, 1], gap="large")

    with col_cm:
        st.markdown('<div class="section-title">Matriz de Confusión</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred_mlp)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Bajo', 'Medio', 'Alto'],
            yticklabels=['Bajo', 'Medio', 'Alto'],
            linewidths=0.5, linecolor='white'
        )
        ax.set_xlabel("Predicción", fontsize=11, fontweight='bold')
        ax.set_ylabel("Valor Real", fontsize=11, fontweight='bold')
        ax.set_title("Matriz de Confusión — MLP", fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_cr:
        st.markdown('<div class="section-title">Reporte por Clase</div>', unsafe_allow_html=True)
        report = classification_report(
            y_test, y_pred_mlp,
            target_names=["Bajo", "Medio", "Alto"],
            output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).T.round(3)
        report_df = report_df.drop(index=["accuracy"], errors="ignore")
        report_df.index = report_df.index.str.replace("macro avg", "Macro Avg").str.replace("weighted avg", "Weighted Avg")
        st.dataframe(
            report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
            use_container_width=True
        )

        st.markdown('<div class="section-title">Distribución Real vs Predicha</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        labels_bar = ["Bajo", "Medio", "Alto"]
        y_test_int = y_test.astype(int)
        real_counts = [sum(y_test_int == i) for i in range(3)]
        pred_counts = [sum(y_pred_mlp.astype(int) == i) for i in range(3)]
        x_pos = np.arange(3)
        bars1 = ax2.bar(x_pos - 0.2, real_counts, 0.35, label="Real",     color=["#22c55e","#f59e0b","#ef4444"], alpha=0.8)
        bars2 = ax2.bar(x_pos + 0.2, pred_counts, 0.35, label="Predicho", color=["#22c55e","#f59e0b","#ef4444"], alpha=0.4, edgecolor=["#22c55e","#f59e0b","#ef4444"], linewidth=1.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels_bar)
        ax2.set_ylabel("Cantidad de pacientes")
        ax2.legend()
        ax2.set_title("Distribución Real vs Predicha", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ══════════════════════════════════════════════
# TAB 3 — COMPARATIVA DE ALGORITMOS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Comparativa: MLP vs Árbol de Decisión (Baseline)</div>', unsafe_allow_html=True)

    acc_dt  = accuracy_score(y_test, y_pred_dt)
    f1m_dt  = f1_score(y_test, y_pred_dt, average="macro", zero_division=0)
    rec_dt  = recall_score(y_test, y_pred_dt, labels=[2], average="macro", zero_division=0)

    comp_df = pd.DataFrame({
        "Algoritmo":     ["Árbol de Decisión (Baseline)", "MLP — Implementado"],
        "Accuracy":      [f"{acc_dt:.3f}", f"{acc:.3f}"],
        "Macro F1":      [f"{f1m_dt:.3f}", f"{f1m:.3f}"],
        "Recall Alto":   [f"{rec_dt:.3f}", f"{rec_h:.3f}"],
        "Estado":        ["📌 Baseline", "✅ Seleccionado"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("**Matriz de Confusión — Árbol de Decisión**")
        cm_dt = confusion_matrix(y_test, y_pred_dt)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges', ax=ax3,
                    xticklabels=['Bajo', 'Medio', 'Alto'],
                    yticklabels=['Bajo', 'Medio', 'Alto'],
                    linewidths=0.5, linecolor='white')
        ax3.set_xlabel("Predicción", fontsize=10)
        ax3.set_ylabel("Valor Real", fontsize=10)
        ax3.set_title("Árbol de Decisión (Baseline)", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_b:
        st.markdown("**Matriz de Confusión — MLP**")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                    xticklabels=['Bajo', 'Medio', 'Alto'],
                    yticklabels=['Bajo', 'Medio', 'Alto'],
                    linewidths=0.5, linecolor='white')
        ax4.set_xlabel("Predicción", fontsize=10)
        ax4.set_ylabel("Valor Real", fontsize=10)
        ax4.set_title("MLP Classifier", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # Gráfico de barras comparativo
    st.markdown("---")
    st.markdown('<div class="section-title">Comparativa Visual de Métricas</div>', unsafe_allow_html=True)
    metrics_names = ["Accuracy", "Macro F1", "Recall Alto"]
    vals_dt  = [acc_dt, f1m_dt, rec_dt]
    vals_mlp = [acc,    f1m,    rec_h]

    fig5, ax5 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics_names))
    ax5.bar(x - 0.2, vals_dt,  0.35, label="Árbol de Decisión", color="#fb923c", alpha=0.85)
    ax5.bar(x + 0.2, vals_mlp, 0.35, label="MLP",               color="#2563eb", alpha=0.85)
    ax5.axhline(0.85, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label="Umbral éxito (0.85)")
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics_names, fontsize=11)
    ax5.set_ylim(0, 1.1)
    ax5.set_ylabel("Score")
    ax5.set_title("Comparativa de Algoritmos — Clasificación de Riesgo", fontsize=12, fontweight='bold')
    ax5.legend()
    for rect, val in zip(ax5.patches, vals_dt + vals_mlp):
        ax5.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    st.markdown("""
    > **Nota metodológica:** Los algoritmos Random Forest y XGBoost están propuestos para la siguiente fase del proyecto.
    > Se espera que superen al MLP actual en las métricas de Recall para la clase Alto, que es la métrica clínica más crítica para ALDIMI.
    """)


# ══════════════════════════════════════════════
# TAB 4 — HISTORIAL
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Historial de Clasificaciones de la Sesión</div>', unsafe_allow_html=True)

    if st.session_state.historial:
        hist_df = pd.DataFrame(st.session_state.historial)

        # KPIs del historial
        total  = len(hist_df)
        altos  = (hist_df["Prioridad"] == "ALTO").sum()
        medios = (hist_df["Prioridad"] == "MEDIO").sum()
        bajos  = (hist_df["Prioridad"] == "BAJO").sum()

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Clasificados", total)
        h2.metric("🔴 Alto Riesgo",  altos,  f"{altos/total*100:.0f}%")
        h3.metric("🟡 Medio Riesgo", medios, f"{medios/total*100:.0f}%")
        h4.metric("🟢 Bajo Riesgo",  bajos,  f"{bajos/total*100:.0f}%")

        if altos > 0:
            st.markdown(f'<div class="alert-box alert-alto">🚨 Hay {altos} paciente(s) de ALTO riesgo en esta sesión. Revisar inmediatamente.</div>', unsafe_allow_html=True)

        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Exportar historial (CSV)",
            data=csv,
            file_name=f"aldimi_historial_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Aún no se han clasificado pacientes en esta sesión. Ve a la pestaña **Clasificación Individual** para comenzar.")
