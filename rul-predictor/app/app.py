import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from preprocess import (
    add_rul, remove_constant_sensors, add_features,
    load_scaler, make_test_last
)

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="RUL Predictor", layout="wide")
st.title("🔧 Previsão de Vida Útil Remanescente (RUL)")
st.write("Envie um arquivo com a estrutura do test_FD001.txt para prever o RUL dos motores.")

# ==========================================
# SIDEBAR (somente upload)
# ==========================================
st.sidebar.title("⚙️ Arquivo de Entrada")
uploaded = st.sidebar.file_uploader("📂 Envie o arquivo", type=["txt"])


# =====================================================================
# PROCESSAMENTO PRINCIPAL
# =====================================================================
if uploaded is not None:

    # ----------------------
    # Leitura
    # ----------------------
    st.info("Lendo arquivo...")
    df = pd.read_csv(uploaded, sep=r"\s+", header=None)

    col_names = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] \
        + [f"s{i}" for i in range(1, 22)]
    df.columns = col_names

    # RUL
    df = add_rul(df, is_test=False)

    # Sensores constantes
    df = remove_constant_sensors(df)

    # Features
    df = add_features(df)

    # Normalização
    scaler = load_scaler("model/scaler.pkl")
    feature_cols = [c for c in df.columns if c not in ["unit_nr", "time_cycles", "RUL"]]
    df[feature_cols] = scaler.transform(df[feature_cols])

    # Última janela
    X_test, units = make_test_last(df, feature_cols, window=30)

    # Modelo
    model = keras.models.load_model("model/model.h5", compile=False)
    preds = model.predict(X_test).flatten()

    # =====================================================================
    # RESULTADOS GERAIS
    # =====================================================================
    st.success("Previsão concluída!")

    res = pd.DataFrame({
        "Motor": units,
        "RUL Previsto": preds
    }).sort_values("RUL Previsto")

    st.subheader("📋 RUL Prevista por Motor (Ordenado)")
    st.dataframe(res)

    # =====================================================================
    # INSIGHTS
    # =====================================================================
    st.subheader("🔍 Insights Automáticos")

    st.write(f"Motor mais crítico: {res.iloc[0]['Motor']} (RUL={res.iloc[0]['RUL Previsto']:.1f})")
    st.write(f"Motor mais saudável: {res.iloc[-1]['Motor']} (RUL={res.iloc[-1]['RUL Previsto']:.1f})")
    st.write(f"RUL médio da frota: {res['RUL Previsto'].mean():.1f}")

    # =====================================================================
    # ZONA DE RISCO
    # =====================================================================
    st.subheader("Distribuição por Zona de Risco")

    riscos = pd.DataFrame({
        "Motor": units,
        "RUL": preds,
        "Risco": np.select(
            [
                preds <= 40,
                (preds > 40) & (preds <= 80),
                preds > 80
            ],
            ["Crítico", "Atenção", "Normal"],
            default="Indefinido"
        )
    }).sort_values("RUL")

    st.dataframe(riscos)

    st.markdown("""
        **Critérios da zona de risco (baseados na RUL prevista):**

        - Crítico: RUL ≤ 40 ciclos  
        - Atenção: 40 < RUL ≤ 80 ciclos  
        - Normal: RUL > 80 ciclos
        """)

    # =====================================================================
    # ANÁLISE INDIVIDUAL POR MOTOR
    # =====================================================================
    st.markdown("---")
    st.header("Análise Individual por Motor")
    st.caption("Selecione o motor e o sensor para visualizar o comportamento histórico.")

    # ----------------------
    # FILTROS ACIMA DO GRÁFICO
    # ----------------------
    col1, col2 = st.columns(2)

    with col1:
        motor_sel = st.selectbox("Selecione um motor:", sorted(df["unit_nr"].unique()))

    with col2:
        sensor_sel = st.selectbox("Selecione um sensor:", [c for c in df.columns if c.startswith("s")])

    # ----------------------
    # GRÁFICO
    # ----------------------
    motor_df = df[df["unit_nr"] == motor_sel].sort_values("time_cycles")

    fig, ax1 = plt.subplots(figsize=(11, 5))

    # SENSOR (azul)
    line1 = ax1.plot(
        motor_df["time_cycles"],
        motor_df[sensor_sel],
        label=f"{sensor_sel} (Valores do sensor)",
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4,
        alpha=0.9
    )
    ax1.set_xlabel("Ciclos de Operação")
    ax1.set_ylabel(sensor_sel, color="#1f77b4")
    ax1.tick_params(axis='y', labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # RUL (vermelho)
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        motor_df["time_cycles"],
        motor_df["RUL"],
        label="RUL (Vida útil remanescente)",
        color="#d62728",
        linewidth=2.2,
        marker="o",
        markersize=4,
        alpha=0.9
    )
    ax2.set_ylabel("RUL", color="#d62728")
    ax2.tick_params(axis='y', labelcolor="#d62728")

    # ----------- LEGENDA -----------
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc="upper center", ncol=2, frameon=False, fontsize=11)

    # ----------- TÍTULO -----------
    plt.title(f"Evolução do Sensor {sensor_sel} e da RUL – Motor {motor_sel}")

    st.pyplot(fig)

    # ----------- TEXTO EXPLICATIVO -----------
    st.markdown("""
        ### 🔍 O que o gráfico representa?

        - **Linha azul** → evolução dos valores do **sensor selecionado** ao longo do tempo.  
        - **Linha vermelha** → **RUL real** (quanto tempo falta até a falha), diminuindo conforme o motor se aproxima do colapso.

        O comportamento conjunto mostra **como a degradação do motor influencia os sensores**.
        """)

    # =====================================================================
    # RUL PREVISTA PARA O MOTOR
    # =====================================================================
    rul_prevista = float(res[res["Motor"] == motor_sel]["RUL Previsto"].values[0])
    st.info(f"RUL prevista pelo modelo para o Motor {motor_sel}: {rul_prevista:.2f} ciclos")
