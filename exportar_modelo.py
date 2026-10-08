"""Entrena el modelo base una vez y guarda los resultados para el dashboard."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

ORIGEN = Path(r"C:\MACHINE_LEARNING\proyecto_magdalena\datos\procesados")
DESTINO = Path("src/datos")
DESTINO.mkdir(parents=True, exist_ok=True)

par = json.loads((ORIGEN / "parametros.json").read_text(encoding="utf-8"))
LIM = tuple(par["LIM"])
TODAS = par["BASE_COLS"] + par["DOM_COLS"]

df = pd.read_csv(ORIGEN / "features.csv", parse_dates=["fecha"])
corte = pd.Timestamp("2010-01-01")
inicio_test = corte + pd.Timedelta(days=30)

dm = df.dropna(subset=["y"])
TR = dm[dm.fecha < corte]
TE = dm[dm.fecha >= inicio_test]
ytr, yte = TR.y.astype(int), TE.y.astype(int)

modelo = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("esc", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
])
modelo.fit(TR[TODAS], ytr)
ys = modelo.predict_proba(TE[TODAS])[:, 1]
yp = modelo.predict(TE[TODAS])

# Persistencia: el estado de hoy se mantiene dentro de 15 dias
yp_per = TE["calamar"].between(*LIM).astype(int).values

filas = []
for nombre, pred, score in [("Dummy", np.zeros(len(yte), int), np.zeros(len(yte))),
                            ("Persistencia", yp_per, yp_per.astype(float)),
                            ("Regresión logística", yp, ys)]:
    filas.append({"modelo": nombre,
                  "accuracy": accuracy_score(yte, pred),
                  "precision": precision_score(yte, pred, zero_division=0),
                  "recall": recall_score(yte, pred, zero_division=0),
                  "f1": f1_score(yte, pred, zero_division=0),
                  "roc_auc": roc_auc_score(yte, score),
                  "pr_auc": average_precision_score(yte, score)})

pd.DataFrame(filas).round(4).to_csv(DESTINO / "metricas.csv", index=False)

# Curvas, submuestreadas para no engordar el archivo
fpr, tpr, _ = roc_curve(yte, ys)
prec, rec, _ = precision_recall_curve(yte, ys)
paso_r = max(1, len(fpr) // 300)
paso_p = max(1, len(prec) // 300)
pd.DataFrame({"fpr": fpr[::paso_r], "tpr": tpr[::paso_r]}).to_csv(DESTINO / "roc.csv", index=False)
pd.DataFrame({"precision": prec[::paso_p], "recall": rec[::paso_p]}).to_csv(DESTINO / "pr.csv", index=False)

cm = confusion_matrix(yte, yp)
pd.DataFrame(cm, index=["real_no", "real_si"], columns=["pred_no", "pred_si"]).to_csv(DESTINO / "confusion.csv")

# Sensibilidad al horizonte
filas = []
for h in (1, 7, 15, 30, 60):
    z = df.copy()
    z["y"] = z["calamar"].between(*LIM).astype(float).shift(-h)
    z = z.dropna(subset=["y"])
    a_, b_ = z[z.fecha < corte], z[z.fecha >= inicio_test]
    yb = b_.y.astype(int)
    m = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("esc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42))])
    m.fit(a_[TODAS], a_.y.astype(int))
    s = m.predict_proba(b_[TODAS])[:, 1]
    per = b_["calamar"].between(*LIM).astype(int).values
    filas.append({"horizonte": h,
                  "modelo": average_precision_score(yb, s),
                  "persistencia": average_precision_score(yb, per)})

pd.DataFrame(filas).round(4).to_csv(DESTINO / "horizonte.csv", index=False)

print("Guardados en src/datos: metricas.csv, roc.csv, pr.csv, confusion.csv, horizonte.csv")
