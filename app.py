import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------

st.set_page_config(
    page_title="ISP Analytics Dashboard",
    page_icon="📡",
    layout="wide"
)

# ------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ------------------------------------------------
# LIMPIEZA BÁSICA
# ------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("📡 Menú Principal")

menu = st.sidebar.radio(
    "Seleccione una opción:",
    [
        "Inicio",
        "Exploración de Datos",
        "KPIs",
        "Visualizaciones",
        "Machine Learning"
    ]
)

# ------------------------------------------------
# INICIO
# ------------------------------------------------

if menu == "Inicio":

    st.title("📡 ISP Analytics Dashboard")

    st.subheader(
        "Predicción de abandono de clientes en empresas proveedoras de Internet"
    )

    st.markdown("---")

    st.write("""
    ### Proyecto Final Integrador

    Esta aplicación permite:

    ✅ Explorar información de clientes ISP  
    ✅ Analizar métricas de abandono (Churn)  
    ✅ Visualizar patrones de comportamiento  
    ✅ Identificar factores asociados a cancelaciones  
    ✅ Apoyar decisiones estratégicas en soporte técnico y atención al cliente
    """)

    st.info("Aplicación creada por Lady Morales Vera")

# ------------------------------------------------
# EXPLORACIÓN DE DATOS
# ------------------------------------------------

elif menu == "Exploración de Datos":

    st.title("📊 Exploración de Datos")

    st.subheader("Vista general del dataset")

    st.dataframe(df)

    st.subheader("Dimensiones del dataset")

    filas, columnas = df.shape

    st.write(f"Filas: {filas}")
    st.write(f"Columnas: {columnas}")

    st.subheader("Tipos de datos")

    st.dataframe(df.dtypes)

    st.subheader("Valores nulos")

    st.dataframe(df.isnull().sum())

# ------------------------------------------------
# KPIs
# ------------------------------------------------

elif menu == "KPIs":

    st.title("📌 Indicadores Clave")

    total_clientes = df.shape[0]

    clientes_churn = df[df["Churn"] == "Yes"].shape[0]

    porcentaje_churn = (
        clientes_churn / total_clientes
    ) * 100

    promedio_facturacion = df["MonthlyCharges"].mean()

    promedio_antiguedad = df["tenure"].mean()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Clientes",
            total_clientes
        )

        st.metric(
            "Clientes que abandonaron",
            clientes_churn
        )

    with col2:

        st.metric(
            "Porcentaje Churn",
            f"{porcentaje_churn:.2f}%"
        )

        st.metric(
            "Promedio Facturación",
            f"${promedio_facturacion:.2f}"
        )

    st.metric(
        "Promedio Antigüedad",
        f"{promedio_antiguedad:.2f} meses"
    )

# ------------------------------------------------
# VISUALIZACIONES
# ------------------------------------------------

elif menu == "Visualizaciones":

    st.title("📈 Visualizaciones")

    contrato = st.selectbox(
        "Seleccione tipo de contrato",
        df["Contract"].unique()
    )

    df_filtrado = df[
        df["Contract"] == contrato
    ]

    # ------------------------------------------------
    # GRÁFICO 1
    # ------------------------------------------------

    fig1 = px.pie(
        df_filtrado,
        names="Churn",
        title="Distribución de Churn"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # ------------------------------------------------
    # GRÁFICO 2
    # ------------------------------------------------

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

    # ------------------------------------------------
    # GRÁFICO 3
    # ------------------------------------------------

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

    # ------------------------------------------------
    # GRÁFICO 4
    # ------------------------------------------------

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
    # ------------------------------------------------
# MACHINE LEARNING
# ------------------------------------------------

# ------------------------------------------------
# MACHINE LEARNING
# ------------------------------------------------

# ------------------------------------------------
# MACHINE LEARNING
# ------------------------------------------------

elif menu == "Machine Learning":

    st.title("🤖 Modelo Predictivo de Churn")

    st.write("""
    Modelo de Machine Learning orientado a la predicción
    de abandono de clientes en empresas proveedoras
    de Internet.
    """)

    # --------------------------------------------
    # COPIA DEL DATASET
    # --------------------------------------------

    df_ml = df.copy()

    # --------------------------------------------
    # ELIMINAR customerID
    # --------------------------------------------

    if "customerID" in df_ml.columns:
        df_ml.drop(
            columns=["customerID"],
            inplace=True
        )

    # --------------------------------------------
    # LIMPIAR TotalCharges
    # --------------------------------------------

    df_ml["TotalCharges"] = pd.to_numeric(
        df_ml["TotalCharges"],
        errors="coerce"
    )

    # --------------------------------------------
    # ELIMINAR NULOS
    # --------------------------------------------

    df_ml.dropna(inplace=True)

    # --------------------------------------------
    # CONVERTIR VARIABLES CATEGÓRICAS
    # --------------------------------------------

    columnas_categoricas = df_ml.select_dtypes(
        include=["object"]
    ).columns

    for col in columnas_categoricas:

        encoder = LabelEncoder()

        df_ml[col] = encoder.fit_transform(
            df_ml[col].astype(str)
        )

    # --------------------------------------------
    # VARIABLES X e y
    # --------------------------------------------

    X = df_ml.drop("Churn", axis=1)

    y = df_ml["Churn"]

    # --------------------------------------------
    # DIVISIÓN TRAIN TEST
    # --------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------
    # MODELO
    # --------------------------------------------

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    # --------------------------------------------
    # PREDICCIONES
    # --------------------------------------------

    y_pred = modelo.predict(X_test)

    # --------------------------------------------
    # ACCURACY
    # --------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.subheader("📌 Precisión del modelo")

    st.success(
        f"Accuracy obtenido: {accuracy:.2f}"
    )

    # --------------------------------------------
    # IMPORTANCIA VARIABLES
    # --------------------------------------------

    importancia = pd.DataFrame({

        "Variable": X.columns,

        "Importancia": modelo.feature_importances_

    })

    importancia = importancia.sort_values(
        by="Importancia",
        ascending=False
    )

    st.subheader("📊 Variables más importantes")

    fig_ml = px.bar(
        importancia.head(10),
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Top 10 variables más importantes"
    )

    st.plotly_chart(
        fig_ml,
        use_container_width=True
    )

    # --------------------------------------------
    # INTERPRETACIÓN
    # --------------------------------------------

    st.info("""
    El modelo Random Forest permite identificar
    factores críticos asociados al abandono
    de clientes, apoyando procesos de soporte
    técnico, fidelización y toma de decisiones
    estratégicas en empresas ISP.
    """)
    # Modelo Random Forest

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    # Predicciones

    y_pred = modelo.predict(X_test)

    # Accuracy

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.subheader("📌 Precisión del modelo")

    st.success(
        f"Accuracy del modelo: {accuracy:.2f}"
    )

    # Importancia de variables

    importancia = pd.DataFrame({

        "Variable": X.columns,

        "Importancia": modelo.feature_importances_

    })

    importancia = importancia.sort_values(
        by="Importancia",
        ascending=False
    )

    st.subheader("📊 Variables más importantes")

    fig_ml = px.bar(
        importancia.head(10),
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Top 10 variables más importantes"
    )

    st.plotly_chart(
        fig_ml,
        use_container_width=True
    )

    st.info("""
    El modelo permite identificar factores asociados al abandono
    de clientes en empresas proveedoras de Internet.
    """)
    # Modelo Random Forest

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    # Predicciones

    y_pred = modelo.predict(X_test)

    # Accuracy

    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("📌 Precisión del modelo")

    st.success(f"Accuracy del modelo: {accuracy:.2f}")

    # Importancia de variables

    importancia = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": modelo.feature_importances_
    })

    importancia = importancia.sort_values(
        by="Importancia",
        ascending=False
    )

    st.subheader("📊 Variables más importantes")

    fig_ml = px.bar(
        importancia.head(10),
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Top 10 variables más importantes"
    )

    st.plotly_chart(
        fig_ml,
        use_container_width=True
    )

    st.info("""
    El modelo permite identificar factores relacionados con el abandono
    de clientes en empresas proveedoras de Internet.
    """)
