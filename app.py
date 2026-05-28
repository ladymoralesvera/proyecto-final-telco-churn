import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------

st.set_page_config(
    page_title="ISP Analytics Dashboard",
    page_icon="📡",
    layout="wide"
)

# ------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------

@st.cache_data
def cargar_datos():
    df = pd.read_csv(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    # Convertir TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df

df = cargar_datos()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("📡 ISP Analytics Dashboard")

menu = st.sidebar.radio(
    "Menú",
    [
        "Inicio",
        "Exploración",
        "Visualizaciones"
    ]
)

# ------------------------------------------------
# INICIO
# ------------------------------------------------

if menu == "Inicio":

    st.title(
        "📡 Predicción de Abandono de Clientes ISP"
    )

    st.markdown("""
    Aplicación desarrollada para analizar
    información de clientes de empresas
    proveedoras de Internet y detectar
    factores asociados al abandono
    del servicio.
    """)

    # KPIs

    total_clientes = len(df)

    clientes_activos = len(
        df[df["Churn"] == "No"]
    )

    clientes_perdidos = len(
        df[df["Churn"] == "Yes"]
    )

    tasa_abandono = (
        clientes_perdidos /
        total_clientes
    ) * 100

    promedio_facturacion = round(
        df["MonthlyCharges"].mean(),
        2
    )

    antiguedad_promedio = round(
        df["tenure"].mean(),
        1
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "👥 Total Clientes",
        total_clientes
    )

    col2.metric(
        "🟢 Clientes Activos",
        clientes_activos
    )

    col3.metric(
        "🔴 Clientes Perdidos",
        clientes_perdidos
    )

    col4, col5 = st.columns(2)

    col4.metric(
        "📉 Tasa de Abandono",
        f"{tasa_abandono:.2f}%"
    )

    col5.metric(
        "💰 Facturación Promedio",
        f"${promedio_facturacion}"
    )

    st.metric(
        "📆 Antigüedad Promedio",
        f"{antiguedad_promedio} meses"
    )

# ------------------------------------------------
# EXPLORACIÓN
# ------------------------------------------------

elif menu == "Exploración":

    st.title("📊 Exploración del Dataset")

    st.subheader("Vista previa")

    st.dataframe(df.head())

    st.subheader("Dimensiones")

    st.write(df.shape)

    st.subheader("Tipos de datos")

    st.dataframe(df.dtypes)

    st.subheader("Valores nulos")

    st.dataframe(df.isnull().sum())

    st.subheader("Estadísticas")

    st.dataframe(df.describe())

# ------------------------------------------------
# VISUALIZACIONES
# ------------------------------------------------

elif menu == "Visualizaciones":

    st.title("📈 Visualizaciones")

    # FILTRO

    contrato = st.selectbox(
        "Seleccione tipo de contrato",
        df["Contract"].unique()
    )

    df_filtrado = df[
        df["Contract"] == contrato
    ]

    # GRÁFICO 1

    fig1 = px.pie(
        df_filtrado,
        names="Churn",
        title="Distribución de Churn"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # GRÁFICO 2

    fig2 = px.histogram(
        df,
        x="InternetService",
        color="Churn",
        title="Churn según Servicio de Internet",
        barmode="group"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # GRÁFICO 3

    fig3 = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Facturación Mensual vs Churn"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # GRÁFICO 4

    fig4 = px.histogram(
        df,
        x="tenure",
        color="Churn",
        title="Antigüedad de Clientes"
    )

st.plotly_chart(
    fig4,
    use_container_width=True
)
