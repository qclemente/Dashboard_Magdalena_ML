"""Calcula una vez los resultados del EDA que el dashboard muestra como tablas.

Se ejecuta con el entorno ml_venv, que tiene scipy y statsmodels:
    C:\\Users\\Qclem\\miniconda3\\envs\\ml_venv\\python.exe exportar_eda.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf, adfuller, kpss

warnings.filterwarnings("ignore")

PROYECTO = Path(r"C:\MACHINE_LEARNING\proyecto_magdalena")
ORIGEN = PROYECTO / "datos" / "procesados"
CRUDOS = PROYECTO / "datos" / "crudos"
DESTINO = Path("src/datos")

par = json.loads((ORIGEN / "parametros.json").read_text(encoding="utf-8"))
LIM = tuple(par["LIM"])
TODAS = par["BASE_COLS"] + par["DOM_COLS"]
CORTE = pd.Timestamp("2010-01-01")
INICIO_TEST = CORTE + pd.Timedelta(days=30)

caudales = pd.read_csv(ORIGEN / "caudales.csv", parse_dates=["fecha"])
train = caudales[caudales.fecha < CORTE]
estaciones = [c for c in caudales.columns if c != "fecha"]


# Auditoria de la serie de transporte: ajuste Qs = a*Q^b por bloques de cinco años
q = pd.read_csv(sorted(CRUDOS.glob("*CALAMAR*Q_MEDIA*.csv"))[0], encoding="utf-8-sig",
                parse_dates=["Fecha"])[["Fecha", "Q_MEDIA_D_m3_s"]]
s = pd.read_csv(sorted(CRUDOS.glob("*CALAMAR*TR_KT*.csv"))[0], encoding="utf-8-sig",
                parse_dates=["Fecha"])[["Fecha", "TR_KT_D_QS_D_kt_dia"]]
par_qs = q.merge(s, on="Fecha").dropna()
par_qs.columns = ["fecha", "Q", "Qs"]
par_qs = par_qs[(par_qs.fecha < CORTE) & (par_qs.Q > 0) & (par_qs.Qs > 0)]

filas = []
for ini, sub in par_qs.groupby(par_qs.fecha.dt.year // 5 * 5):
    if len(sub) < 150:
        continue
    b, loga, r, _, _ = st.linregress(np.log10(sub.Q), np.log10(sub.Qs))
    filas.append({"Bloque": f"{ini}-{ini + 4}", "Días": len(sub),
                  "Exponente b": round(b, 4), "R²": round(r ** 2, 6)})
pd.DataFrame(filas).to_csv(DESTINO / "auditoria_transporte.csv", index=False)


# Mecanismo de los faltantes: caudal en Calamar los dias con y sin dato en cada estacion
filas = []
for e in estaciones:
    if e == "calamar":
        continue
    ini, fin = caudales.loc[caudales[e].notna(), "fecha"].agg(["min", "max"])
    sub = caudales[(caudales.fecha >= ini) & (caudales.fecha <= fin)]
    con = sub.loc[sub[e].notna(), "calamar"].dropna()
    sin = sub.loc[sub[e].isna(), "calamar"].dropna()
    if len(sin) < 30:
        continue
    u, p = st.mannwhitneyu(con, sin)
    filas.append({"Estación": e, "Huecos internos": len(sin),
                  "Mediana con dato": round(con.median()), "Mediana sin dato": round(sin.median()),
                  "Efecto A": round(u / (len(con) * len(sin)), 3)})
pd.DataFrame(filas).to_csv(DESTINO / "faltantes_mecanismo.csv", index=False)


# Estacionariedad y autocorrelacion de la estacion objetivo
serie = train["calamar"].dropna()
ac = acf(serie.values, nlags=90, fft=True)
estacionariedad = {
    "adf_p": float(adfuller(serie, autolag="AIC")[1]),
    "kpss_p": float(kpss(serie, regression="c", nlags="auto")[1]),
    "acf": [float(v) for v in ac],
}
(DESTINO / "estacionariedad.json").write_text(json.dumps(estacionariedad), encoding="utf-8")


# Auditoria de fuga: AUC univariado de cada predictor
df = pd.read_csv(ORIGEN / "features.csv", parse_dates=["fecha"])
dtr = df[df.fecha < CORTE].dropna(subset=["y"])
filas = []
for c in TODAS:
    sub = dtr[[c, "y"]].dropna()
    if len(sub) < 500:
        continue
    auc = roc_auc_score(sub.y, sub[c])
    filas.append({"Variable": c, "AUC univariado": round(max(auc, 1 - auc), 4)})
(pd.DataFrame(filas).sort_values("AUC univariado", ascending=False).head(10)
 .to_csv(DESTINO / "fuga.csv", index=False))


# Verificacion del desempeño: accuracy y PR-AUC segun el horizonte
def modelo():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("esc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                                random_state=42))])


filas = []
for h in (1, 7, 15, 30, 60):
    z = df.copy()
    z["y"] = z["calamar"].between(*LIM).astype(float).shift(-h)
    z = z.dropna(subset=["y"])
    a, b = z[z.fecha < CORTE], z[z.fecha >= INICIO_TEST]
    yb = b.y.astype(int)
    m = modelo().fit(a[TODAS], a.y.astype(int))
    persist = b["calamar"].between(*LIM).astype(int)
    filas.append({"Horizonte (días)": h,
                  "Accuracy modelo": round(accuracy_score(yb, m.predict(b[TODAS])), 3),
                  "Accuracy persistencia": round(accuracy_score(yb, persist), 3),
                  "Accuracy dummy": round(1 - yb.mean(), 3),
                  "PR-AUC modelo": round(average_precision_score(yb, m.predict_proba(b[TODAS])[:, 1]), 3),
                  "PR-AUC persistencia": round(average_precision_score(yb, persist), 3)})
pd.DataFrame(filas).to_csv(DESTINO / "verificacion.csv", index=False)

print("Guardados: auditoria_transporte.csv, faltantes_mecanismo.csv, estacionariedad.json, "
      "fuga.csv, verificacion.csv")
