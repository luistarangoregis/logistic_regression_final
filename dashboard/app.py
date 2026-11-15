"""
Dashboard: Predicción del riesgo de enfermedad coronaria a 10 años (Framingham)
Modelo: Regresión Logística

Ejecutar desde la carpeta raíz del proyecto:
    python -m streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# ----------------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Riesgo Coronario · Framingham", page_icon="🫀", layout="wide")

RAIZ = Path(__file__).resolve().parent.parent

# Paleta (identidad de las dos clases + estados de riesgo)
AZUL = "#2a78d6"        # Sin CHD / modelo final
NARANJA = "#eb6834"     # Con CHD
GRIS = "#8c8b86"        # referencia / modelo base
TEXTO_2 = "#52514e"
GRID = "#e6e5e0"
ESTADO = {"bajo": "#0ca30c", "elevado": "#fab219", "muy_elevado": "#d03b3b"}

NOMBRES = {
    "male": "Sexo masculino", "age": "Edad", "education": "Educación", "currentSmoker": "Fumador actual",
    "cigsPerDay": "Cigarrillos por día", "BPMeds": "Medicamento para la presión", "prevalentStroke": "Derrame previo",
    "prevalentHyp": "Hipertensión", "diabetes": "Diabetes", "totChol": "Colesterol total (mg/dL)",
    "sysBP": "Presión sistólica (mmHg)", "diaBP": "Presión diastólica (mmHg)", "BMI": "IMC (kg/m²)",
    "heartRate": "Frecuencia cardiaca (lpm)", "glucose": "Glucosa (mg/dL)", "TenYearCHD": "CHD a 10 años",
}
NOMBRES_OR = {
    "prevalentStroke": "Derrame previo", "age": "Edad (+10 años)", "male": "Sexo masculino",
    "sysBP": "Presión sistólica (+20 mmHg)", "prevalentHyp": "Hipertensión",
    "cigsPerDay": "Cigarrillos/día (+10)", "glucose": "Glucosa (+20 mg/dL)", "totChol": "Colesterol (+40 mg/dL)",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.9rem;}
      .nota {color: #52514e; font-size: 0.9rem;}
      .resultado {border-radius: 12px; padding: 1.1rem 1.3rem; border: 1px solid #e6e5e0; background: #ffffff;}
      .resultado h2 {margin: 0.2rem 0 0.4rem 0;}
      .etiqueta {display: inline-block; padding: 0.15rem 0.7rem; border-radius: 999px; font-weight: 600;
                 font-size: 0.95rem; color: #0b0b0b;}
      .paso {text-align: center; padding: 0.6rem 0.3rem; border-radius: 10px; background: #f1f0ec;
             font-size: 0.88rem; line-height: 1.25;}
      .paso b {display: block; font-size: 0.95rem; margin-bottom: 0.15rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Carga de artefactos generados por el notebook
# ----------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load(RAIZ / "models" / "modelo_regresion_logistica.pkl")


@st.cache_data
def cargar_datos():
    datos = pd.read_csv(RAIZ / "data" / "processed" / "framingham_limpio.csv")
    predicciones = pd.read_csv(RAIZ / "data" / "processed" / "predicciones_prueba.csv", index_col=0)
    with open(RAIZ / "models" / "metricas.json", encoding="utf-8") as f:
        metricas = json.load(f)
    return datos, predicciones, metricas


try:
    artefacto = cargar_modelo()
    datos, predicciones, metricas = cargar_datos()
except FileNotFoundError:
    st.error("No se encontraron los archivos del modelo. Ejecuta primero el notebook "
             "`Proyecto_Regresion_Logistica_CHD.ipynb` completo.")
    st.stop()

modelo = artefacto["pipeline"]
UMBRAL = float(artefacto["umbral"])
VARIABLES = artefacto["variables"]
TASA_BASE = datos["TenYearCHD"].mean()


def estilo(fig, alto=360, leyenda=True):
    """Estilo común para todas las gráficas de Plotly."""
    fig.update_layout(
        template="plotly_white", height=alto, margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#0b0b0b"),
        title_font=dict(size=15), showlegend=leyenda, plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
        hoverlabel=dict(bgcolor="white", font_size=13), bargap=0.3, bargroupgap=0.08,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=TEXTO_2))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=TEXTO_2))
    return fig


def nivel_riesgo(p):
    if p < UMBRAL:
        return "bajo", "✅ Riesgo bajo"
    if p < 0.30:
        return "elevado", "⚠️ Riesgo elevado"
    return "muy_elevado", "⛔ Riesgo muy elevado"


def metricas_umbral(real, prob, umbral):
    pred = (prob >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(real, pred, labels=[0, 1]).ravel()
    sens = vp / (vp + fn) if (vp + fn) else 0
    esp = vn / (vn + fp) if (vn + fp) else 0
    prec = vp / (vp + fp) if (vp + fp) else 0
    exac = (vp + vn) / len(real)
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
    return dict(vn=vn, fp=fp, fn=fn, vp=vp, sens=sens, esp=esp, prec=prec, exac=exac, f1=f1)


# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
st.title("🫀 Riesgo de enfermedad coronaria a 10 años")
st.markdown(
    "<span class='nota'>Modelo de **Regresión Logística** entrenado con datos del *Framingham Heart Study* "
    f"({len(datos):,} pacientes) · Ingeniería del Conocimiento</span>",
    unsafe_allow_html=True,
)

tab_resumen, tab_eda, tab_modelo, tab_calc = st.tabs(
    ["📋 Resumen", "🔎 Exploración de datos", "📈 Desempeño del modelo", "🧮 Calculadora de riesgo"]
)

# ============================================================================
# 1. RESUMEN
# ============================================================================
with tab_resumen:
    m_final = metricas["prueba"]
    m_base = metricas["base"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes analizados", f"{len(datos):,}", border=True)
    c2.metric("Desarrollaron CHD en 10 años", f"{TASA_BASE:.1%}", border=True,
              help="Proporción de pacientes con enfermedad coronaria a 10 años (clase minoritaria).")
    c3.metric("Sensibilidad del modelo", f"{m_final['Sensibilidad (recall)']:.1%}",
              delta=f"{(m_final['Sensibilidad (recall)'] - m_base['Sensibilidad (recall)']) * 100:+.1f} pp vs. modelo base",
              border=True, help="De los pacientes que sí desarrollaron CHD, porcentaje que el modelo detecta.")
    c4.metric("ROC-AUC (prueba)", f"{m_final['ROC-AUC']:.3f}", border=True,
              help="Capacidad del modelo para ordenar a los pacientes por riesgo (0.5 = azar, 1 = perfecto).")

    st.subheader("Ciclo de datos del proyecto")
    pasos = [
        ("1. Obtención", "Framingham Heart Study<br>4,238 pacientes"),
        ("2. Extracción", "Carga y revisión<br>de calidad"),
        ("3. Limpieza", "Imputación por tipo<br>de variable"),
        ("4. EDA", "Factores asociados<br>al riesgo"),
        ("5. Modelo", "Regresión Logística<br>8 variables"),
        ("6. Evaluación", f"Umbral {UMBRAL:.2f}<br>AUC {m_final['ROC-AUC']:.3f}"),
        ("7. Visualización", "Este dashboard"),
    ]
    for col, (titulo, desc) in zip(st.columns(len(pasos)), pasos):
        col.markdown(f"<div class='paso'><b>{titulo}</b>{desc}</div>", unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns([1.1, 1])
    with izq:
        st.subheader("Hallazgos principales")
        st.markdown(
            f"""
- **La edad es el factor más fuerte:** el riesgo pasa de **4 %** (30–39 años) a **28 %** (60–70 años).
- **Hipertensión, diabetes y derrame previo** multiplican el riesgo entre **2.3 y 2.9 veces**.
- **Los hombres** tienen **1.5 veces** más riesgo que las mujeres.
- El modelo detecta a **{m_final['Sensibilidad (recall)']:.0%}** de los pacientes que desarrollarán CHD,
  frente a **{m_base['Sensibilidad (recall)']:.0%}** del modelo base.
- **Factores modificables** con mayor efecto: presión sistólica, tabaquismo y glucosa.
"""
        )
        st.info("Herramienta de **tamizaje** con fines académicos: prioriza a quién evaluar con más detalle; "
                "no sustituye un diagnóstico médico.", icon="ℹ️")
    with der:
        grupos = pd.cut(datos["age"], bins=[30, 39, 49, 59, 70], labels=["30–39", "40–49", "50–59", "60–70"])
        por_edad = datos.groupby(grupos, observed=True)["TenYearCHD"].agg(["mean", "count"]).reset_index()
        fig = go.Figure(go.Bar(
            x=por_edad["age"].astype(str), y=por_edad["mean"] * 100, marker_color=NARANJA,
            marker=dict(cornerradius=4), text=[f"{v:.1f}%" for v in por_edad["mean"] * 100], textposition="outside",
            customdata=por_edad["count"],
            hovertemplate="Edad %{x}<br>%{y:.1f}% desarrolló CHD<br>%{customdata:,} pacientes<extra></extra>",
        ))
        fig.update_layout(title="% de pacientes con CHD a 10 años por grupo de edad",
                          yaxis_title="% con CHD", xaxis_title="Edad (años)")
        fig.update_yaxes(range=[0, por_edad["mean"].max() * 100 * 1.25])
        st.plotly_chart(estilo(fig, 340, leyenda=False), width="stretch", config={"displayModeBar": False})

# ============================================================================
# 2. EXPLORACIÓN DE DATOS
# ============================================================================
with tab_eda:
    st.markdown("<span class='nota'>Filtra la población; todas las gráficas e indicadores de esta pestaña se "
                "actualizan.</span>", unsafe_allow_html=True)
    f1, f2, f3, f4, f5 = st.columns([1, 1.6, 1, 1, 1])
    f_sexo = f1.selectbox("Sexo", ["Todos", "Mujeres", "Hombres"])
    f_edad = f2.slider("Edad", int(datos["age"].min()), int(datos["age"].max()),
                       (int(datos["age"].min()), int(datos["age"].max())))
    f_fuma = f3.selectbox("Fumador", ["Todos", "Sí", "No"])
    f_hip = f4.selectbox("Hipertensión", ["Todos", "Sí", "No"])
    f_dia = f5.selectbox("Diabetes", ["Todos", "Sí", "No"])

    filtro = datos["age"].between(*f_edad)
    if f_sexo != "Todos":
        filtro &= datos["male"] == (1 if f_sexo == "Hombres" else 0)
    for valor, col in [(f_fuma, "currentSmoker"), (f_hip, "prevalentHyp"), (f_dia, "diabetes")]:
        if valor != "Todos":
            filtro &= datos[col] == (1 if valor == "Sí" else 0)
    df = datos[filtro]

    if len(df) < 10:
        st.warning("La selección tiene muy pocos pacientes; los resultados no son representativos. "
                   "Amplía los filtros.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pacientes en la selección", f"{len(df):,}", border=True)
    dif = (df["TenYearCHD"].mean() - TASA_BASE) * 100
    k2.metric("% con CHD a 10 años", f"{df['TenYearCHD'].mean():.1%}",
              delta=None if len(df) == len(datos) else f"{dif:+.1f} pp vs. población total",
              delta_color="inverse", border=True)
    k3.metric("Edad promedio", f"{df['age'].mean():.1f} años", border=True)
    k4.metric("Presión sistólica promedio", f"{df['sysBP'].mean():.0f} mmHg", border=True)

    g1, g2 = st.columns(2)
    with g1:
        binarias = {"male": "Sexo<br>masculino", "currentSmoker": "Fumador<br>actual",
                    "BPMeds": "Medicamento<br>presión", "prevalentStroke": "Derrame<br>previo",
                    "prevalentHyp": "Hipertensión", "diabetes": "Diabetes"}
        filas = []
        for col, corto in binarias.items():
            for v, etiqueta in [(0, "No"), (1, "Sí")]:
                sub = df[df[col] == v]
                if len(sub):
                    filas.append({"factor": corto, "presenta": etiqueta,
                                  "tasa": sub["TenYearCHD"].mean() * 100, "n": len(sub)})
        tb = pd.DataFrame(filas)
        fig = go.Figure()
        for etiqueta, color in [("No", AZUL), ("Sí", NARANJA)]:
            s = tb[tb["presenta"] == etiqueta]
            fig.add_bar(x=s["factor"], y=s["tasa"], name=f"{etiqueta} presenta el factor", marker_color=color,
                        marker=dict(cornerradius=4), customdata=s["n"],
                        hovertemplate="%{x}<br>" + etiqueta + ": %{y:.1f}% con CHD<br>%{customdata:,} pacientes<extra></extra>")
        fig.update_layout(title="% con CHD según cada factor", barmode="group", yaxis_title="% con CHD")
        st.plotly_chart(estilo(fig, 380), width="stretch", config={"displayModeBar": False})

    with g2:
        var = st.selectbox("Variable numérica", ["age", "sysBP", "diaBP", "totChol", "glucose", "BMI",
                                                 "cigsPerDay", "heartRate"], format_func=lambda c: NOMBRES[c])
        fig = go.Figure()
        for clase, nombre, color in [(0, "Sin CHD", AZUL), (1, "Con CHD", NARANJA)]:
            fig.add_box(y=df.loc[df["TenYearCHD"] == clase, var], name=nombre, marker_color=color,
                        boxmean=True, line=dict(width=1.5))
        fig.update_layout(title=f"{NOMBRES[var]}: pacientes sin y con CHD", yaxis_title=NOMBRES[var])
        st.plotly_chart(estilo(fig, 320, leyenda=False), width="stretch", config={"displayModeBar": False})

    with st.expander("Ver datos de la selección"):
        st.dataframe(df.drop(columns=["conjunto"]).rename(columns=NOMBRES), width="stretch", height=300)

# ============================================================================
# 3. DESEMPEÑO DEL MODELO
# ============================================================================
with tab_modelo:
    real = predicciones["real"].values
    prob = predicciones["probabilidad"].values

    st.markdown(
        f"Resultados sobre el **conjunto de prueba** ({len(real)} pacientes que el modelo nunca vio, "
        f"{int(real.sum())} con CHD). Mueve el **umbral de decisión** para ver el equilibrio entre detectar más "
        f"casos y generar más falsas alarmas. El umbral elegido en el notebook es **{UMBRAL:.2f}**."
    )
    umbral = st.slider("Umbral de decisión", 0.02, 0.80, UMBRAL, 0.01)
    m = metricas_umbral(real, prob, umbral)
    ref = metricas_umbral(real, prob, UMBRAL)

    def delta(v, r):
        return None if abs(v - r) < 1e-9 else f"{(v - r) * 100:+.1f} pp vs. {UMBRAL:.2f}"

    a, b, c, d, e = st.columns(5)
    a.metric("Sensibilidad", f"{m['sens']:.1%}", delta(m["sens"], ref["sens"]), border=True,
             help="Pacientes con CHD detectados / total de pacientes con CHD")
    b.metric("Especificidad", f"{m['esp']:.1%}", delta(m["esp"], ref["esp"]), border=True,
             help="Pacientes sin CHD identificados como sanos / total sin CHD")
    c.metric("Precisión", f"{m['prec']:.1%}", delta(m["prec"], ref["prec"]), border=True,
             help="De los marcados como riesgo, % que realmente desarrolló CHD")
    d.metric("Exactitud", f"{m['exac']:.1%}", delta(m["exac"], ref["exac"]), border=True)
    e.metric("ROC-AUC", f"{roc_auc_score(real, prob):.3f}", border=True, help="No depende del umbral")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        z = [[m["vn"], m["fp"]], [m["fn"], m["vp"]]]
        textos = [[f"<b>{m['vn']}</b><br>Sanos bien identificados", f"<b>{m['fp']}</b><br>Falsas alarmas"],
                  [f"<b>{m['fn']}</b><br>Casos NO detectados", f"<b>{m['vp']}</b><br>Casos detectados"]]
        # Color por significado (aciertos en azul, errores en naranja), no por cantidad
        fig = go.Figure(go.Heatmap(
            z=[[1, 0], [0, 1]], x=["Predicción: sin CHD", "Predicción: con CHD"], y=["Real: sin CHD", "Real: con CHD"],
            text=textos, texttemplate="%{text}", textfont=dict(size=14, color="#0b0b0b"), customdata=z,
            colorscale=[[0, "#fbe1d6"], [1, "#d5e5f8"]], showscale=False, xgap=4, ygap=4,
            hovertemplate="%{y} · %{x}<br>%{customdata} pacientes<extra></extra>",
        ))
        fig.update_layout(title=f"Matriz de confusión (umbral {umbral:.2f}) · azul = acierto, naranja = error")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(estilo(fig, 380, leyenda=False), width="stretch", config={"displayModeBar": False})

    with col_roc:
        fpr, tpr, _ = roc_curve(real, prob)
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS, dash="dash", width=1.5),
                        name="Azar (AUC 0.5)", hoverinfo="skip")
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=AZUL, width=2.5),
                        name=f"Modelo (AUC {roc_auc_score(real, prob):.3f})",
                        hovertemplate="1 - especificidad: %{x:.2f}<br>Sensibilidad: %{y:.2f}<extra></extra>")
        fig.add_scatter(x=[1 - m["esp"]], y=[m["sens"]], mode="markers", name=f"Umbral {umbral:.2f}",
                        marker=dict(size=13, color=NARANJA, line=dict(color="white", width=2)),
                        hovertemplate="Umbral actual<br>Sensibilidad %{y:.1%}<extra></extra>")
        fig.update_layout(title="Curva ROC", xaxis_title="Tasa de falsos positivos (1 − especificidad)",
                          yaxis_title="Sensibilidad")
        st.plotly_chart(estilo(fig, 380), width="stretch", config={"displayModeBar": False})

    col_u, col_cal = st.columns(2)
    with col_u:
        us = np.arange(0.02, 0.81, 0.01)
        curvas = pd.DataFrame([metricas_umbral(real, prob, u) | {"u": u} for u in us])
        fig = go.Figure()
        for col, nombre, color in [("sens", "Sensibilidad", NARANJA), ("esp", "Especificidad", AZUL),
                                   ("prec", "Precisión", "#1baf7a")]:
            fig.add_scatter(x=curvas["u"], y=curvas[col], mode="lines", name=nombre, line=dict(color=color, width=2),
                            hovertemplate=nombre + ": %{y:.1%}<extra></extra>")
        fig.add_vline(x=umbral, line=dict(color="#0b0b0b", width=1.5, dash="dot"))
        fig.update_layout(title="Métricas según el umbral", xaxis_title="Umbral", yaxis_title="Valor",
                          hovermode="x unified")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(estilo(fig, 360), width="stretch", config={"displayModeBar": False})

    with col_cal:
        cortes = [0, 0.05, 0.10, 0.20, 0.30, 1.0]
        etiquetas = ["< 5 %", "5–10 %", "10–20 %", "20–30 %", "> 30 %"]
        estr = pd.DataFrame({"e": pd.cut(prob, cortes, labels=etiquetas), "r": real}) \
            .groupby("e", observed=False)["r"].agg(["mean", "count"]).reset_index()
        fig = go.Figure(go.Bar(
            x=estr["e"].astype(str), y=estr["mean"] * 100, marker_color=NARANJA, marker=dict(cornerradius=4),
            text=[f"{v:.0f}%" for v in estr["mean"] * 100], textposition="outside", customdata=estr["count"],
            hovertemplate="Riesgo estimado %{x}<br>%{y:.1f}% desarrolló CHD<br>%{customdata} pacientes<extra></extra>",
        ))
        fig.update_layout(title="¿Las probabilidades son confiables? (% real de CHD por nivel de riesgo)",
                          xaxis_title="Probabilidad estimada por el modelo", yaxis_title="% real con CHD")
        fig.update_yaxes(range=[0, estr["mean"].max() * 100 * 1.3])
        st.plotly_chart(estilo(fig, 360, leyenda=False), width="stretch", config={"displayModeBar": False})

    col_or, col_tab = st.columns([1.2, 1])
    with col_or:
        orr = pd.DataFrame(metricas["odds_ratios"]).sort_values("odds ratio")
        orr["nombre"] = orr["variable"].map(NOMBRES_OR)
        signif = orr["valor p"] < 0.05
        fig = go.Figure(go.Scatter(
            x=orr["odds ratio"], y=orr["nombre"], mode="markers",
            marker=dict(size=11, color=np.where(signif, NARANJA, GRIS), line=dict(color="white", width=2)),
            error_x=dict(type="data", symmetric=False, array=orr["IC 95% superior"] - orr["odds ratio"],
                         arrayminus=orr["odds ratio"] - orr["IC 95% inferior"], color=GRIS, thickness=1.5, width=5),
            customdata=np.stack([orr["IC 95% inferior"], orr["IC 95% superior"], orr["valor p"]], axis=1),
            hovertemplate="%{y}<br>Odds ratio: ×%{x:.2f}<br>IC 95%: %{customdata[0]:.2f} – %{customdata[1]:.2f}"
                          "<br>valor p: %{customdata[2]:.3f}<extra></extra>",
        ))
        fig.add_vline(x=1, line=dict(color="#0b0b0b", width=1, dash="dash"))
        fig.update_xaxes(type="log", title="Odds ratio (escala logarítmica) — > 1 aumenta el riesgo",
                         tickvals=[0.8, 1, 1.5, 2, 3, 5, 10], ticktext=["0.8", "1", "1.5", "2", "3", "5", "10"])
        fig.update_layout(title="Factores de riesgo (naranja = estadísticamente significativo)")
        st.plotly_chart(estilo(fig, 380, leyenda=False), width="stretch", config={"displayModeBar": False})

    with col_tab:
        st.markdown("**Modelo base vs. modelo final (conjunto de prueba)**")
        comp = pd.DataFrame({"Modelo base (umbral 0.5)": metricas["base"],
                             f"Modelo final (umbral {UMBRAL:.2f})": metricas["prueba"]})
        st.dataframe(comp.style.format("{:.1%}", subset=pd.IndexSlice[comp.index != "ROC-AUC", :])
                     .format("{:.3f}", subset=pd.IndexSlice[["ROC-AUC"], :]), width="stretch")
        st.markdown(
            "<span class='nota'>El modelo base parece más exacto, pero solo detecta a 1 de cada 13 pacientes "
            "con CHD. El modelo final prioriza la <b>sensibilidad</b>: en tamizaje es preferible una falsa alarma "
            "(estudios adicionales) que un caso no detectado.</span>", unsafe_allow_html=True)

# ============================================================================
# 4. CALCULADORA DE RIESGO
# ============================================================================
PERFILES = {
    "Mujer joven saludable": dict(c_edad=38, c_sexo="Mujer", c_fuma=False, c_cigs=0, c_sys=112, c_glu=80,
                                  c_col=185, c_hip=False, c_der=False),
    "Hombre de 55 años, fumador": dict(c_edad=55, c_sexo="Hombre", c_fuma=True, c_cigs=20, c_sys=138, c_glu=88,
                                       c_col=240, c_hip=False, c_der=False),
    "Adulto mayor con varios factores": dict(c_edad=66, c_sexo="Hombre", c_fuma=True, c_cigs=25, c_sys=172,
                                             c_glu=150, c_col=285, c_hip=True, c_der=False),
}
for k, v in PERFILES["Hombre de 55 años, fumador"].items():
    st.session_state.setdefault(k, v)


def cargar_perfil(nombre):
    for k, v in PERFILES[nombre].items():
        st.session_state[k] = v


def predecir(edad, hombre, cigs, sys_bp, glu, col, hip, der):
    x = pd.DataFrame([{"male": hombre, "age": edad, "cigsPerDay": cigs, "sysBP": sys_bp, "glucose": glu,
                       "prevalentStroke": der, "prevalentHyp": hip, "totChol": col}])
    return float(modelo.predict_proba(x[VARIABLES])[0, 1])


with tab_calc:
    st.markdown("<span class='nota'>Captura los datos del paciente o carga un perfil de ejemplo. "
                "El modelo usa las 8 variables seleccionadas en el notebook.</span>", unsafe_allow_html=True)
    b1, b2, b3, _ = st.columns([1, 1, 1, 1.2])
    for col_b, nombre in zip([b1, b2, b3], PERFILES):
        col_b.button(nombre, on_click=cargar_perfil, args=(nombre,), width="stretch")

    entrada, salida = st.columns([1, 1.15], gap="large")
    with entrada:
        with st.container(border=True):
            st.markdown("**Datos del paciente**")
            i1, i2 = st.columns(2)
            edad = i1.slider("Edad (años)", 30, 75, key="c_edad")
            sexo = i2.radio("Sexo", ["Mujer", "Hombre"], horizontal=True, key="c_sexo")
            fuma = i1.toggle("Fuma actualmente", key="c_fuma")
            cigs = i2.slider("Cigarrillos por día", 0, 60, key="c_cigs", disabled=not fuma)
            sys_bp = i1.number_input("Presión sistólica (mmHg)", 80, 300, step=1, key="c_sys",
                                     help="Normal: menos de 120 mmHg")
            glu = i2.number_input("Glucosa (mg/dL)", 40, 400, step=1, key="c_glu",
                                  help="Normal en ayuno: 70 – 100 mg/dL")
            col = i1.number_input("Colesterol total (mg/dL)", 100, 700, step=1, key="c_col",
                                  help="Deseable: menos de 200 mg/dL")
            i2.write("")
            hip = i2.toggle("Diagnóstico de hipertensión", key="c_hip")
            der = i2.toggle("Ha sufrido un derrame cerebral", key="c_der")

    cigs_ef = cigs if fuma else 0
    hombre = int(sexo == "Hombre")
    p = predecir(edad, hombre, cigs_ef, sys_bp, glu, col, int(hip), int(der))
    clave, texto_nivel = nivel_riesgo(p)

    with salida:
        st.markdown(
            f"""
            <div class='resultado'>
              <span class='nota'>Probabilidad estimada de enfermedad coronaria en 10 años</span>
              <h2 style='font-size:3rem'>{p:.1%}</h2>
              <span class='etiqueta' style='background:{ESTADO[clave]}33; border:1.5px solid {ESTADO[clave]}'>
                {texto_nivel}</span>
              <p class='nota' style='margin-top:0.7rem'>
                Equivale a <b>{p / TASA_BASE:.1f} veces</b> el riesgo promedio de la población ({TASA_BASE:.1%}).
                Umbral de alerta del modelo: {UMBRAL:.0%}.
                {"<br><b>Recomendación:</b> evaluación cardiológica y control de factores de riesgo." if p >= UMBRAL else ""}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Comparación con la población y con personas de su mismo sexo y grupo de edad
        similares = datos[(datos["male"] == hombre) & datos["age"].between(edad - 5, edad + 5)]
        comp = pd.DataFrame({
            "grupo": ["Este paciente", f"{'Hombres' if hombre else 'Mujeres'} de {edad - 5}–{edad + 5} años",
                      "Población total"],
            "valor": [p * 100, similares["TenYearCHD"].mean() * 100, TASA_BASE * 100],
        }).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=comp["valor"], y=comp["grupo"], orientation="h", marker=dict(cornerradius=4),
            marker_color=[GRIS, GRIS, ESTADO[clave]], text=[f"{v:.1f}%" for v in comp["valor"]],
            textposition="outside", hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig.update_xaxes(range=[0, max(comp["valor"].max() * 1.3, 20)], ticksuffix="%")
        fig.update_layout(title="Comparación del riesgo")
        st.plotly_chart(estilo(fig, 200, leyenda=False), width="stretch", config={"displayModeBar": False})

    # ¿Cuánto bajaría el riesgo al controlar cada factor modificable?
    escenarios = []
    if cigs_ef > 0:
        escenarios.append(("Dejar de fumar", dict(cigs=0)))
    if sys_bp > 120:
        escenarios.append((f"Presión sistólica a 120 mmHg (actual {sys_bp})", dict(sys_bp=120)))
    if glu > 100:
        escenarios.append((f"Glucosa a 90 mg/dL (actual {glu})", dict(glu=90)))
    if col > 200:
        escenarios.append((f"Colesterol a 200 mg/dL (actual {col})", dict(col=200)))

    st.subheader("¿Cuánto bajaría el riesgo controlando los factores modificables?")
    if not escenarios:
        st.success("Los factores modificables de este paciente ya están en rangos saludables. "
                   "Su riesgo se explica por factores no modificables (edad, sexo, antecedentes).", icon="✅")
    else:
        base_args = dict(edad=edad, hombre=hombre, cigs=cigs_ef, sys_bp=sys_bp, glu=glu, col=col,
                         hip=int(hip), der=int(der))
        filas = []
        for nombre, cambio in escenarios:
            filas.append((nombre, predecir(**(base_args | cambio))))
        todos = base_args | {k: v for _, c in escenarios for k, v in c.items()}
        filas.append(("Todos los anteriores", predecir(**todos)))
        esc = pd.DataFrame(filas, columns=["escenario", "prob"])
        esc["reduccion"] = (p - esc["prob"]) * 100

        ce1, ce2 = st.columns([1.4, 1])
        with ce1:
            esc_g = esc.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=esc_g["prob"] * 100, y=esc_g["escenario"], orientation="h",
                marker_color=[AZUL if e != "Todos los anteriores" else "#1c5cab" for e in esc_g["escenario"]],
                marker=dict(cornerradius=4), text=[f"{v:.1f}%" for v in esc_g["prob"] * 100], textposition="outside",
                hovertemplate="%{y}<br>Riesgo: %{x:.1f}%<extra></extra>",
            ))
            fig.add_vline(x=p * 100, line=dict(color=NARANJA, width=2, dash="dash"),
                          annotation_text=f"Riesgo actual {p:.1%}", annotation_position="top",
                          annotation_font_color=TEXTO_2)
            fig.update_layout(title=None, xaxis_title="Riesgo estimado (%)")
            fig.update_xaxes(range=[0, max(p * 100 * 1.25, 5)])
            st.plotly_chart(estilo(fig, 90 + 48 * len(esc), leyenda=False), width="stretch", config={"displayModeBar": False})
        with ce2:
            total = esc.iloc[-1]
            st.metric("Riesgo controlando todos los factores", f"{total['prob']:.1%}",
                      delta=f"{-total['reduccion']:.1f} pp", delta_color="inverse", border=True)
            st.markdown("<span class='nota'>Simulación con el mismo modelo, cambiando solo el factor indicado y "
                        "manteniendo constantes los demás. Es una estimación estadística, no una promesa "
                        "clínica.</span>", unsafe_allow_html=True)

    st.caption("⚕️ Herramienta académica de tamizaje basada en datos del Framingham Heart Study. "
               "No sustituye la valoración de un profesional de la salud.")
