import numpy as np
import pandas as pd
import joblib

# ============================================================
# FUNÇÃO OFICIAL DO S-SCORE (PHM08)
# ============================================================
def s_score(y_true, y_pred):
    d = y_pred - y_true
    s = np.where(
        d < 0, np.exp(-d / 13) - 1,   # subestimativa
        np.exp(d / 10) - 1           # superestimativa
    )
    return np.sum(s)

# ============================================================
# RUL COM LIMITE (125)
# ============================================================
def add_rul(df, rul_dict=None, is_test=False, rul_cap=125):
    df = df.copy()

    # calcula ciclos máximos por motor
    m = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    m.columns = ["unit_nr", "max_cycle"]
    df = df.merge(m, on="unit_nr", how="left")

    df["RUL"] = df["max_cycle"] - df["time_cycles"]

    # NO APP — nunca usa RUL_FD001
    df["RUL"] = df["RUL"].clip(upper=rul_cap)
    df.drop("max_cycle", axis=1, inplace=True)

    return df

# ============================================================
# REMOVER SENSORES CONSTANTES (listagem real)
# ============================================================

CONSTANT_COLUMNS = ['setting_3', 's1', 's5', 's10', 's16', 's18', 's19']

def remove_constant_sensors(df):
    df = df.drop(columns=[c for c in CONSTANT_COLUMNS if c in df.columns])
    return df

# ============================================================
# FEATURE ENGINEERING (mean5, std5, slope)
# ============================================================
def add_features(df):
    df = df.copy()
    sensor_cols = [c for c in df.columns if c.startswith("s")]

    for s in sensor_cols:
        df[f"{s}_mean5"] = (
            df.groupby("unit_nr")[s]
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        df[f"{s}_std5"] = (
            df.groupby("unit_nr")[s]
            .rolling(5, min_periods=1)
            .std()
            .reset_index(0, drop=True)
            .fillna(0)
        )

        df[f"{s}_slope"] = df.groupby("unit_nr")[s].diff().fillna(0)

    return df

# ============================================================
# CARREGAR SCALER
# ============================================================
def load_scaler(path):
    return joblib.load(path)

# ============================================================
# ÚLTIMA JANELA
# ============================================================
def make_test_last(df, feature_cols, window=30):
    X = []
    units = []

    for u in df["unit_nr"].unique():
        unit_df = df[df["unit_nr"] == u]
        feat = unit_df[feature_cols].values
        X.append(feat[-window:])
        units.append(u)

    return np.array(X), units
