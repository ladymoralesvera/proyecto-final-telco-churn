import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración
st.set_page_config(
    page_title="ISP Analytics Dashboard",
    page_icon="📡",
    layout="wide"
)

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = cargar_datos()

# Sidebar
st.sidebar.title("📡 ISP Analytics Dashboard")

menu = st.sidebar.radio(
    "Seleccione una opción",
    [
        "Inicio",
        "Exploración de Datos",
        "Visualizaciones"
    ]
)

# --------------------------
# INICIO
# --------------------------

if menu == "Inicio":

    st.title("📡 Predicción de Abandono de Clientes ISP")

    st.markdown("""
    Aplicación desarrollada para analizar el comportamiento
    de clientes de empresas proveedoras de Internet y detectar
    factores asociados al abandono del servicio.
    """)

    total_clientes = len(df)

    clientes_activos = len(df[df["Churn"] == "No"])

    clientes_perdidos = len(df[df["Churn"] == "Yes"])

    tasa_abandono = (clientes_perdidos / total_clientes) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Clientes", total_clientes)
    col2.metric("Clientes Activos", clientes_activos)
    col3.metric("Clientes Perdidos", clientes_perdidos)
    col4.metric("Tasa de Abandono", f"{tasa_abandono:.2f}%")

# --------------------------
# EXPLORACIÓN
# --------------------------

elif menu == "Exploración de Datos":

    st.title("📊 Exploración del Dataset")

    st.subheader("Vista previa")

    st.dataframe(df.head())

    st.subheader("Dimensiones")

    st.write(df.shape)

    st.subheader("Tipos de datos")

    st.dataframe(df.dtypes)

    st.subheader("Valores nulos")

    st.dataframe(df.isnull().sum())

# --------------------------
# VISUALIZACIONES
# --------------------------

elif menu == "Visualizaciones":

    st.title("📈 Análisis Visual")

    churn_count = df["Churn"].value_counts()

    fig1 = px.pie(
        values=churn_count.values,
        names=churn_count.index,
        title="Distribución de Clientes Activos y Perdidos"
    )

    st.plotly_chart(fig1, use_container_width=True)

    contrato = px.histogram(
        df,
        x="Contract",
        color="Churn",
        title="Abandono según Tipo de Contrato"
    )

    st.plotly_chart(contrato, use_container_width=True)
