import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
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

st.sidebar.image(
    "https://images.unsplash.com/photo-1516321318423-f06f85e504b3",
    use_container_width=True
)

st.sidebar.title("📡 ISP Analytics")

st.sidebar.markdown("""
Sistema de análisis predictivo para empresas proveedoras de Internet.
""")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Seleccione una opción:",
[
    "Inicio",
    "Exploración de Datos",
    "KPIs",
    "Visualizaciones",
    "Machine Learning",
    "Predicción",
    "Acerca del Proyecto"
]
)

# ------------------------------------------------
# INICIO
# ------------------------------------------------

if menu == "Inicio":

    st.title("📡 ISP Analytics Dashboard")
    st.image(
    "https://images.unsplash.com/photo-1558494949-ef010cbdcc31",
    use_container_width=True
    )    

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

    st.title("📊 Visualizaciones")

    # --------------------------------------------
    # CHURN
    # --------------------------------------------

    st.subheader("Clientes con abandono")

    fig1 = px.histogram(
        df,
        x="Churn",
        color="Churn",
        title="Distribución de abandono"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # --------------------------------------------
    # CONTRATOS
    # --------------------------------------------

    st.subheader("Tipos de contrato")

    fig2 = px.histogram(
        df,
        x="Contract",
        color="Contract",
        title="Distribución por contrato"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # --------------------------------------------
    # CARGOS MENSUALES
    # --------------------------------------------

    st.subheader("Facturación mensual")

    fig3 = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Facturación mensual vs abandono"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # --------------------------------------------
    # ANTIGÜEDAD
    # --------------------------------------------

    st.subheader("Antigüedad del cliente")

    fig4 = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=30,
        title="Antigüedad de clientes"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # --------------------------------------------
    # MATRIZ DE CORRELACIÓN
    # --------------------------------------------

    st.subheader("📌 Matriz de Correlación")

    try:

        df_corr = df.copy()

        # Convertir TotalCharges

        df_corr["TotalCharges"] = pd.to_numeric(
            df_corr["TotalCharges"],
            errors="coerce"
        )

        df_corr.dropna(inplace=True)

        # Codificar columnas categóricas

        columnas_categoricas = df_corr.select_dtypes(
            include=["object"]
        ).columns

        for col in columnas_categoricas:

            encoder = LabelEncoder()

            df_corr[col] = encoder.fit_transform(
                df_corr[col].astype(str)
            )

        correlacion = df_corr.corr(numeric_only=True)

        fig_corr, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            correlacion,
            cmap="Blues",
            ax=ax
        )

        st.pyplot(fig_corr)

    except Exception as e:

        st.error(f"Error en correlación: {e}")
# ------------------------------------------------
# MACHINE LEARNING
# ------------------------------------------------

elif menu == "Machine Learning":

    st.title("🤖 Modelo Predictivo de Churn")

    st.write("""
    Modelo de Machine Learning orientado a la predicción
    de abandono de clientes en empresas proveedoras de Internet.
    """)

    try:

        # ----------------------------------------
        # COPIA DEL DATASET
        # ----------------------------------------

        df_ml = df.copy()

        # ----------------------------------------
        # ELIMINAR customerID
        # ----------------------------------------

        if "customerID" in df_ml.columns:

            df_ml.drop(
                columns=["customerID"],
                inplace=True
            )

        # ----------------------------------------
        # LIMPIAR TotalCharges
        # ----------------------------------------

        df_ml["TotalCharges"] = pd.to_numeric(
            df_ml["TotalCharges"],
            errors="coerce"
        )

        # ----------------------------------------
        # ELIMINAR NULOS
        # ----------------------------------------

        df_ml.dropna(inplace=True)

        # ----------------------------------------
        # CODIFICAR VARIABLES
        # ----------------------------------------

        columnas_categoricas = df_ml.select_dtypes(
            include=["object"]
        ).columns

        for col in columnas_categoricas:

            encoder = LabelEncoder()

            df_ml[col] = encoder.fit_transform(
                df_ml[col].astype(str)
            )

        # ----------------------------------------
        # VARIABLES
        # ----------------------------------------

        X = df_ml.drop("Churn", axis=1)

        y = df_ml["Churn"]

        # ----------------------------------------
        # TRAIN TEST
        # ----------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ----------------------------------------
        # MODELO
        # ----------------------------------------

        modelo = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        modelo.fit(X_train, y_train)

        # ----------------------------------------
        # PREDICCIONES
        # ----------------------------------------

        y_pred = modelo.predict(X_test)

        # ----------------------------------------
        # ACCURACY
        # ----------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        st.subheader("📌 Precisión del Modelo")

        st.success(
            f"Accuracy obtenido: {accuracy:.2f}"
        )

        # ----------------------------------------
        # IMPORTANCIA VARIABLES
        # ----------------------------------------

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
        El modelo Random Forest permite identificar
        patrones asociados al abandono de clientes
        en empresas proveedoras de Internet.
        """)

    except Exception as e:

        st.error(f"Error detectado: {e}")

# ------------------------------------------------
# PREDICCIÓN
# ------------------------------------------------

elif menu == "Predicción":

    st.title("📡 Predicción de Abandono")

    st.write("""
    Simulación académica de predicción
    de abandono de clientes.
    """)

    genero = st.selectbox(
        "Género",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Adulto mayor",
        [0, 1]
    )

    tenure = st.slider(
        "Antigüedad del cliente",
        1,
        72,
        12
    )

    monthly = st.slider(
        "Facturación mensual",
        20,
        150,
        70
    )

    contrato = st.selectbox(
        "Tipo de contrato",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet = st.selectbox(
        "Servicio de Internet",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    if st.button("Predecir abandono"):

        if contrato == "Month-to-month" and monthly > 80:

            st.error(
                "⚠️ Alta probabilidad de abandono"
            )

        else:

            st.success(
                "✅ Baja probabilidad de abandono"
            )

# ------------------------------------------------
# ACERCA DEL PROYECTO
# ------------------------------------------------

elif menu == "Acerca del Proyecto":

    st.title("📘 Acerca del Proyecto")

    st.markdown("""

    ## Proyecto Final Integrador

    Aplicación desarrollada en Streamlit
    para análisis de clientes ISP y
    predicción de abandono utilizando
    Machine Learning.

    ### Tecnologías utilizadas

    - Python
    - Streamlit
    - Pandas
    - Plotly
    - Scikit-learn
    - Random Forest

    ### Área de aplicación

    Telecomunicaciones, soporte técnico
    y atención al cliente en empresas
    proveedoras de Internet.

    ### Objetivo

    Identificar patrones relacionados
    con abandono de clientes para
    mejorar procesos de fidelización
    y soporte operativo.

    ### Autor

    Lady Morales Vera
    Ingeniería en Telecomunicaciones

    """)
 
