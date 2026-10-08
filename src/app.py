"""
Dashboard: descarga efectiva en el río Magdalena
Kevin Clemente Rosario y Manuel Meza Castro
"""
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# ---------------------------------------------------------------
# Datos: se cargan una sola vez, al arrancar la app
# ---------------------------------------------------------------
DATOS = Path(__file__).parent / "datos"

caudales = pd.read_csv(DATOS / "caudales.csv", parse_dates=["fecha"])
ubicaciones = pd.read_csv(DATOS / "ubicaciones.csv", sep=";", encoding="utf-8-sig")
metricas = pd.read_csv(DATOS / "metricas.csv")
roc = pd.read_csv(DATOS / "roc.csv")
pr = pd.read_csv(DATOS / "pr.csv")
confusion = pd.read_csv(DATOS / "confusion.csv", index_col=0)
auditoria = pd.read_csv(DATOS / "auditoria_transporte.csv")
mecanismo = pd.read_csv(DATOS / "faltantes_mecanismo.csv")
fuga = pd.read_csv(DATOS / "fuga.csv")
verificacion = pd.read_csv(DATOS / "verificacion.csv")
estacionariedad = json.loads((DATOS / "estacionariedad.json").read_text(encoding="utf-8"))
par = json.loads((DATOS / "parametros.json").read_text(encoding="utf-8"))

Q_EF = par["Q_EF"]
LIM = par["LIM"]
TRANSITOS = par["TRANSITOS"]
ESTACIONES = [c for c in caudales.columns if c != "fecha"]
CORTE = pd.Timestamp("2010-01-01")

app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True,
           title="Descarga efectiva - Río Magdalena")
server = app.server


# ---------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------
def fig_mapa():
    fig = px.scatter_map(
        ubicaciones, lat="Latitud", lon="Longitud",
        hover_name="Estacion",
        hover_data={"Municipio": True, "Departamento": True,
                    "Altitud (m s. n. m.)": True, "Latitud": False, "Longitud": False},
        color="Altitud (m s. n. m.)", color_continuous_scale="Turbo",
        zoom=4.6, map_style="open-street-map", height=460)
    fig.update_traces(marker=dict(size=14))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


def fig_faltantes():
    d = (100 * caudales[ESTACIONES].isna().mean()).sort_values().reset_index()
    d.columns = ["estacion", "faltante"]
    fig = px.bar(d, x="faltante", y="estacion", orientation="h", text_auto=".1f",
                 labels={"faltante": "Días sin dato (%)", "estacion": ""})
    fig.update_layout(height=360, margin=dict(t=10))
    return fig


def fig_cobertura():
    anual = caudales.set_index("fecha")[ESTACIONES].notna().resample("YS").mean() * 100
    fig = px.imshow(anual.T, aspect="auto", color_continuous_scale="Blues",
                    x=anual.index.year, labels={"x": "Año", "y": "", "color": "% con dato"})
    fig.update_layout(height=360, margin=dict(t=10))
    return fig


def fig_magnitud_frecuencia():
    serie = caudales.loc[caudales.fecha < CORTE, "calamar"].dropna()
    bordes = np.logspace(np.log10(serie.min()), np.log10(serie.max()), 26)
    frec, _ = np.histogram(serie, bins=bordes)
    centros = np.sqrt(bordes[:-1] * bordes[1:])
    carga = frec * par["A_OF"] * centros ** par["B_OF"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=centros, y=100 * frec / frec.sum(), name="Frecuencia (% de días)",
                         marker_color="steelblue", opacity=0.6))
    fig.add_trace(go.Scatter(x=centros, y=100 * carga / carga.sum(),
                             name="Carga transportada (%)", mode="lines+markers",
                             line=dict(color="sienna", width=3)))
    fig.add_vline(x=Q_EF, line_dash="dash", line_color="red",
                  annotation_text=f"Q_ef = {Q_EF:,.0f} m³/s")
    fig.update_layout(xaxis_title="Clase de caudal (m³/s)", yaxis_title="Porcentaje",
                      height=430, margin=dict(t=30), legend=dict(orientation="h", y=1.1))
    return fig


def fig_estacional():
    d = caudales.copy()
    d["mes"] = d.fecha.dt.month
    fig = px.box(d, x="mes", y="calamar",
                 labels={"mes": "Mes", "calamar": "Caudal (m³/s)"})
    fig.add_hline(y=Q_EF, line_dash="dash", line_color="red",
                  annotation_text=f"Q_ef = {Q_EF:,.0f} m³/s")
    fig.update_layout(height=400, margin=dict(t=30))
    return fig


def fig_duracion():
    fig = go.Figure()
    for e in ESTACIONES:
        s = caudales[e].dropna().sort_values(ascending=False).values
        p = 100 * (pd.Series(range(1, len(s) + 1)) / (len(s) + 1))
        paso = max(1, len(s) // 600)
        fig.add_trace(go.Scatter(x=p[::paso], y=s[::paso], name=e, mode="lines"))
    fig.add_hrect(y0=LIM[0], y1=LIM[1], fillcolor="red", opacity=0.12, line_width=0)
    fig.update_layout(xaxis_title="Probabilidad de excedencia (%)",
                      yaxis_title="Caudal (m³/s)", yaxis_type="log",
                      height=400, margin=dict(t=30))
    return fig


def fig_acf():
    ac = estacionariedad["acf"]
    fig = px.bar(x=list(range(len(ac))), y=ac,
                 labels={"x": "Rezago (días)", "y": "Autocorrelación"})
    for k in (15, 30):
        fig.add_vline(x=k, line_dash="dot", line_color="gray",
                      annotation_text=f"{k} d: {ac[k]:.2f}")
    fig.update_layout(height=380, margin=dict(t=30), yaxis_range=[0, 1.05])
    return fig


def fig_correlacion():
    m = caudales[ESTACIONES].corr().round(2)
    fig = px.imshow(m, text_auto=True, color_continuous_scale="RdYlBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=430, margin=dict(t=10), coloraxis_showscale=False)
    return fig


def fig_transito():
    # El nombre del catálogo no coincide con la clave de la serie, se mapea a mano
    MAPA = {"ARRANCAPLUMAS - AUT": "arrancaplumas", "PUERTO SALGAR - AUT": "pto_salgar",
            "TRES CRUCES": "tres_cruces", "COYONGAL": "coyongal",
            "EL BANCO - AUT": "el_banco", "SAN ROQUE": "san_roque", "CALAMAR": "calamar"}
    u = ubicaciones.assign(estacion=ubicaciones.Estacion.map(MAPA))
    d = (pd.DataFrame({"estacion": list(TRANSITOS), "dias": list(TRANSITOS.values())})
         .merge(u, on="estacion", how="left")
         .dropna(subset=["Altitud (m s. n. m.)"])
         .sort_values("dias"))
    fig = px.scatter(d, x="Altitud (m s. n. m.)", y="dias", text="estacion",
                     labels={"dias": "Tiempo de tránsito (días)"})
    fig.update_traces(textposition="top center", marker=dict(size=13, color="sienna"))
    fig.update_layout(height=430, margin=dict(t=10))
    return fig


def fig_roc():
    fig = px.line(roc, x="fpr", y="tpr",
                  labels={"fpr": "Tasa de falsos positivos",
                          "tpr": "Tasa de verdaderos positivos"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="gray"))
    fig.update_layout(height=400, margin=dict(t=10))
    return fig


def fig_pr():
    fig = px.line(pr, x="recall", y="precision",
                  labels={"recall": "Exhaustividad", "precision": "Precisión"})
    prevalencia = confusion.loc["real_si"].sum() / confusion.values.sum()
    fig.add_hline(y=prevalencia, line_dash="dash", line_color="gray",
                  annotation_text=f"azar ({prevalencia:.2f})")
    fig.update_layout(height=400, margin=dict(t=10))
    return fig


def fig_confusion():
    fig = px.imshow(confusion.values, text_auto=True, color_continuous_scale="Blues",
                    x=["predicho: no", "predicho: sí"], y=["real: no", "real: sí"])
    fig.update_layout(height=400, margin=dict(t=10), coloraxis_showscale=False)
    return fig


def fig_horizonte():
    d = verificacion.melt(id_vars="Horizonte (días)",
                          value_vars=["PR-AUC modelo", "PR-AUC persistencia"],
                          var_name="serie", value_name="PR-AUC")
    fig = px.line(d, x="Horizonte (días)", y="PR-AUC", color="serie", markers=True,
                  labels={"serie": ""})
    fig.update_layout(height=400, margin=dict(t=10))
    return fig


# ---------------------------------------------------------------
# Componentes de presentación
# ---------------------------------------------------------------
def tarjeta(titulo, valor):
    return dbc.Card(dbc.CardBody([
        html.H6(titulo, className="text-muted mb-1"),
        html.H4(valor, className="mb-0"),
    ]))


def parrafo(texto):
    return html.P(texto, style={"textAlign": "justify"})


def nota(texto):
    return html.P(texto, className="text-muted small", style={"textAlign": "justify"})


def tabla(df):
    return dbc.Table.from_dataframe(df, striped=True, bordered=False,
                                    hover=True, size="sm", className="mt-2")


def bloque(titulo, figura, interpretacion):
    return dbc.Card(dbc.CardBody([
        html.H5(titulo),
        dcc.Graph(figure=figura),
        nota(interpretacion),
    ]), className="mb-4")


def seccion(titulo, *contenido):
    """Sección desplegable dentro de una pestaña."""
    return dbc.AccordionItem(list(contenido), title=titulo)


def acordeon(*secciones):
    return dbc.Accordion(list(secciones), start_collapsed=True, always_open=True,
                         className="mb-4")


# ---------------------------------------------------------------
# Pestaña 1: contexto del problema
# ---------------------------------------------------------------
def tab_contexto():
    return dbc.Container([

        dbc.Row([
            dbc.Col(tarjeta("Descarga efectiva", f"{Q_EF:,.0f} m³/s"), md=3),
            dbc.Col(tarjeta("Banda de transporte dominante",
                            f"{LIM[0]:,.0f} – {LIM[1]:,.0f} m³/s"), md=3),
            dbc.Col(tarjeta("Periodo de registro",
                            f"{caudales.fecha.min():%Y} – {caudales.fecha.max():%Y}"), md=3),
            dbc.Col(tarjeta("Observaciones diarias", f"{len(caudales):,}"), md=3),
        ], className="mb-4 g-3"),

        acordeon(

            seccion("Pregunta de investigación",
                html.Blockquote(
                    "¿Es posible clasificar, con la información hidrológica disponible hasta "
                    "el día t, si el día t+15 pertenecerá al rango de descarga efectiva del río "
                    "Magdalena en la estación Calamar?",
                    className="blockquote border-start border-4 ps-3 mb-0"),
            ),

            seccion("Planteamiento del problema",
                parrafo(
                    "El río Magdalena drena cerca de la cuarta parte del territorio colombiano y "
                    "presenta uno de los rendimientos sedimentarios más altos del mundo por "
                    "unidad de área drenada, rasgo que se atribuye a la combinación de relieve "
                    "pronunciado, actividad tectónica y precipitación intensa en la cordillera "
                    "(Milliman y Syvitski, 1992). El sedimento que transporta condiciona la "
                    "forma del cauce, la vida útil de los embalses y la profundidad disponible "
                    "para la navegación."),
                parrafo(
                    "El transporte acumulado a lo largo de los años depende del producto entre "
                    "la cantidad de sedimento que moviliza cada caudal y la frecuencia con que "
                    "ese caudal se presenta. Dado que la primera crece con el caudal mientras "
                    "que la segunda decrece, el producto alcanza su máximo en un valor "
                    "intermedio del rango, denominado descarga efectiva, que se asocia al "
                    "caudal responsable de la configuración geométrica del cauce (Wolman y "
                    "Miller, 1960)."),
                parrafo(
                    "La descarga efectiva caracteriza el comportamiento sedimentario de largo "
                    "plazo, aunque su valor no indica cuándo volverá a presentarse. La operación "
                    "de embalses y la programación de dragados en el canal navegable requieren "
                    "esa anticipación, por lo que este trabajo evalúa si el caudal registrado "
                    "hasta el día t en siete estaciones del cauce permite predecir si el caudal "
                    "del día t+15 en Calamar caerá dentro de la banda de descarga efectiva. La "
                    "tarea se formula, en consecuencia, como un problema de clasificación "
                    "binaria."),
            ),

            seccion("Datos y delimitación",
                parrafo(
                    "Se emplean las series de caudal medio diario de siete estaciones del IDEAM, "
                    "descargadas del portal DHIME, cuyo registro en la estación objetivo abarca "
                    "de 1940 a 2026. El estudio se limita al cauce principal del Magdalena; el "
                    "aporte del Cauca y de los demás afluentes se manifiesta de forma indirecta "
                    "a través del caudal de las estaciones del bajo Magdalena. El modelo se "
                    "ajusta con la información anterior a 2010 y se evalúa sobre los dieciséis "
                    "años posteriores. La licencia del portal autoriza la descarga para uso "
                    "personal y no comercial, de modo que los archivos originales no se "
                    "redistribuyen."),
                dcc.Graph(figure=fig_mapa()),
                nota(
                    "El gradiente altitudinal, que va de 222 m s. n. m. en Guaduas a 8 m en "
                    "Calamar, ordena las estaciones a lo largo del cauce. Ese orden se emplea en "
                    "la pestaña de análisis exploratorio para contrastar los tiempos de tránsito "
                    "estimados a partir de las series de caudal."),
                tabla(ubicaciones),
            ),
        ),

    ], fluid=True)


# ---------------------------------------------------------------
# Pestaña 2: análisis exploratorio
# ---------------------------------------------------------------
def tab_eda():
    exp_b = auditoria["Exponente b"]
    ac = estacionariedad["acf"]
    return dbc.Container([

        acordeon(

            seccion("Calidad de los datos",
                dbc.Row([
                    dbc.Col(bloque("Días sin dato por estación", fig_faltantes(),
                        "Calamar conserva el 98,4 % del registro, mientras que las estaciones "
                        "del bajo Magdalena carecen de dato entre el 39 y el 48 % de los "
                        "días."), md=6),
                    dbc.Col(bloque("Cobertura anual de la red", fig_cobertura(),
                        "Cada estación presenta un bloque de ausencia anterior a su "
                        "instalación, ocurrida entre 1972 y 1975, de modo que la mayor parte de "
                        "los faltantes refleja la construcción progresiva de la red."), md=6),
                ]),
                parrafo(
                    "Dado que la ausencia depende de la fecha, una variable observada, el "
                    "mecanismo dominante es MAR. Para descartar un componente MNAR se comparó, "
                    "dentro del periodo operativo de cada estación, el caudal de Calamar en los "
                    "días con y sin dato mediante la prueba de Mann-Whitney. Ningún tamaño de "
                    "efecto A cae por debajo de 0,42, valor que indicaría huecos concentrados en "
                    "aguas altas; en Coyongal y Puerto Salgar los huecos se concentran en aguas "
                    "bajas, situación que no sesga la estimación de la descarga efectiva. La "
                    "imputación por mediana dentro del Pipeline resulta, por lo tanto, "
                    "adecuada."),
                tabla(mecanismo),
            ),

            seccion("Auditoría de la serie de transporte",
                parrafo(
                    "El campo de metadatos de los archivos de transporte de sedimentos declara "
                    "que el valor se obtiene con la curva EQ_QS-QL. Para verificar si la serie "
                    "aporta información independiente se ajustó la relación Qs = a·Q^b por "
                    "bloques de cinco años sobre el periodo de entrenamiento."),
                tabla(auditoria),
                parrafo(
                    f"Seis de los ocho bloques alcanzan un R² superior a 0,999, y el exponente "
                    f"pasa de {exp_b.iloc[0]:.2f} hasta 1984 a {exp_b.iloc[4]:.2f} en la década "
                    f"de 1990 y a {exp_b.iloc[-1]:.2f} desde 2000. La relación resulta exacta "
                    "dentro de cada periodo y el IDEAM modificó la ecuación varias veces, de "
                    "modo que la serie es una transformación determinista del caudal. Por esa "
                    "razón se excluyó tanto de los predictores como de la variable objetivo, y "
                    "la curva vigente se utilizó únicamente para localizar la descarga "
                    "efectiva."),
            ),

            seccion("Descarga efectiva y variable objetivo",
                bloque("Análisis magnitud-frecuencia en Calamar, 1940-2009",
                       fig_magnitud_frecuencia(),
                    "El registro se agrupa en 25 clases logarítmicas. La frecuencia decrece "
                    "hacia los caudales altos, mientras que la carga transportada, producto de "
                    "esa frecuencia por la magnitud que asigna la curva oficial, alcanza su "
                    "máximo en una clase intermedia. La descarga efectiva resultante, 9.954 "
                    "m³/s, equivale a 1,38 veces el caudal medio y se excede el 15,4 % del "
                    "tiempo. Las cuatro ecuaciones históricas del IDEAM conducen a la misma "
                    "clase modal."),
                parrafo(
                    "La variable objetivo toma el valor 1 cuando el caudal del día t+15 cae "
                    f"dentro de la banda de ±15 % alrededor de la descarga efectiva, entre "
                    f"{LIM[0]:,.0f} y {LIM[1]:,.0f} m³/s. Se prefirió la banda a la excedencia "
                    "porque el concepto se refiere a un rango de caudales en torno al máximo de "
                    "transporte. En el conjunto de entrenamiento la clase positiva representa "
                    "el 25,7 % de los días, con un desbalance de 2,9 a 1, de modo que la "
                    "evaluación prioriza el F1 y el área bajo la curva precisión-exhaustividad "
                    "sobre el accuracy."),
            ),

            seccion("Series de caudal y régimen hidrológico",
                dbc.Card(dbc.CardBody([
                    html.H5("Serie de caudal"),
                    nota("Seleccione una o varias estaciones. La banda roja marca el rango de "
                         "descarga efectiva y las series se agregan por mes."),
                    dcc.Dropdown(id="sel-estaciones",
                                 options=[{"label": e, "value": e} for e in ESTACIONES],
                                 value=["calamar"], multi=True),
                    dcc.Graph(id="g-serie"),
                ]), className="mb-4"),
                dbc.Row([
                    dbc.Col(bloque("Régimen estacional en Calamar", fig_estacional(),
                        "El régimen es bimodal, con mínimo en febrero y marzo, máximo principal "
                        "en noviembre y diciembre y un máximo secundario a mitad de año, "
                        "asociado al doble paso de la Zona de Convergencia Intertropical. El "
                        "modelo lo incorpora mediante componentes cíclicos anual y "
                        "semianual."), md=6),
                    dbc.Col(bloque("Curvas de duración de caudales", fig_duracion(),
                        "Los caudales crecen de forma monótona aguas abajo. Calamar presenta "
                        "una distribución casi simétrica, con asimetría de 0,26, porque la "
                        "cuenca baja amortigua los picos generados aguas arriba; por esa razón "
                        "no se aplicó transformación logarítmica."), md=6),
                ]),
            ),

            seccion("Dependencia temporal",
                bloque("Autocorrelación del caudal diario en Calamar", fig_acf(),
                    f"La autocorrelación alcanza {ac[1]:.3f} a un día, {ac[15]:.2f} a quince y "
                    f"{ac[30]:.2f} a treinta. Las pruebas ADF (p < 0,001) y KPSS (p = 0,10), "
                    "cuyas hipótesis nulas son opuestas, coinciden en que la serie es "
                    "estacionaria en nivel."),
                parrafo(
                    "Una dependencia de esta magnitud implica que el tamaño efectivo de muestra "
                    "es muy inferior al número de filas y que una partición aleatoria produciría "
                    "una estimación optimista del desempeño. Por esa razón la partición es "
                    "cronológica, con corte en 2010 y un intervalo de separación de treinta "
                    "días, y la validación cruzada emplea TimeSeriesSplit."),
            ),

            seccion("Relación entre estaciones",
                dbc.Row([
                    dbc.Col(bloque("Correlación entre estaciones", fig_correlacion(),
                        "Arrancaplumas y Puerto Salgar correlacionan fuertemente entre sí y "
                        "débilmente con Calamar, mientras que las estaciones del bajo Magdalena "
                        "lo hacen con la estación objetivo. Entre ambos grupos ingresan los "
                        "aportes del Cauca y el Cesar y se interpone la depresión momposina, "
                        "que amortigua la onda de crecida."), md=6),
                    dbc.Col(bloque("Tránsito de la onda de crecida", fig_transito(),
                        "El rezago que maximiza la correlación con Calamar va de 7 días en "
                        "Coyongal a 21 en Puerto Salgar. Dado que se estimó sin información "
                        "geográfica, su correlación con la altitud (Spearman de 0,96) y con la "
                        "latitud (Pearson de −0,88) constituye una verificación externa, y cada "
                        "estación entra al modelo con su rezago."), md=6),
                ]),
            ),

            seccion("Auditoría de fuga de datos",
                parrafo(
                    "Todos los predictores usan información disponible hasta el día t, mientras "
                    "que el objetivo se sitúa en t+15. Como control adicional se calculó el área "
                    "bajo la curva ROC de cada predictor por separado, dado que un valor cercano "
                    "a 1 delataría una variable derivada del objetivo. La tabla recoge las diez "
                    "variables con mayor AUC."),
                tabla(fuga),
                parrafo(
                    "Ninguna variable alcanza el umbral de alerta de 0,95. La distancia a la "
                    "descarga efectiva llega a 0,93 por construcción, puesto que mide la "
                    "cercanía del caudal de hoy al centro de la banda, y su poder predictivo "
                    "decae con el horizonte."),
            ),
        ),

    ], fluid=True)


# ---------------------------------------------------------------
# Pestaña 3: modelos base
# ---------------------------------------------------------------
def tab_modelos():
    return dbc.Container([

        acordeon(

            seccion("Modelo base y referencias",
                parrafo(
                    "El modelo base es una regresión logística con ponderación equilibrada de "
                    "clases, integrada en un Pipeline con imputación por mediana y escalado "
                    "estándar, de modo que ambas transformaciones se ajustan solo con el "
                    "conjunto de entrenamiento. Se compara con dos referencias: un clasificador "
                    "trivial que predice siempre la clase mayoritaria, y la persistencia, que "
                    "supone que dentro de quince días el río estará en la misma condición que "
                    "hoy. Dada la autocorrelación de la serie, la persistencia constituye la "
                    "referencia exigente."),
                html.H5("Desempeño en el conjunto de prueba, 2010-2026", className="mt-3"),
                tabla(metricas),
            ),

            seccion("Verificación del desempeño",
                parrafo(
                    "Un accuracy de 0,882 obliga a verificar que el resultado no proviene de "
                    "fuga de datos, de un problema trivial o del desbalance de clases. La fuga "
                    "se descartó en la auditoría del análisis exploratorio. El clasificador "
                    "trivial alcanza 0,751 sin capacidad predictiva alguna, y la persistencia "
                    "obtiene 0,883, de modo que el accuracy no distingue al modelo de una regla "
                    "que se limita a repetir el estado actual. Por esa razón la evaluación se "
                    "apoya en el PR-AUC, donde el modelo obtiene 0,872 frente a 0,642 de la "
                    "persistencia."),
                tabla(verificacion),
                parrafo(
                    "Alargar el horizonte reduce el accuracy del modelo, que a treinta días cae "
                    "a 0,766, aunque a ese horizonte queda por debajo de la persistencia en esa "
                    "misma métrica. La ventaja del modelo en PR-AUC alcanza su máximo a quince "
                    "días, con una diferencia de 0,230 frente a 0,154 a siete días y 0,190 a "
                    "treinta, lo que justifica el horizonte adoptado."),
            ),

            seccion("Curvas de evaluación",
                dbc.Row([
                    dbc.Col(bloque("Curva precisión-exhaustividad", fig_pr(),
                        "Con una clase positiva del 25 %, esta curva resulta más informativa "
                        "que la ROC porque se centra en la clase minoritaria. La línea "
                        "horizontal marca el desempeño de una predicción aleatoria."), md=6),
                    dbc.Col(bloque("Curva ROC", fig_roc(),
                        "El área bajo la curva alcanza 0,959. La diagonal corresponde a un "
                        "clasificador sin información."), md=6),
                ]),
                dbc.Row([
                    dbc.Col(bloque("Matriz de confusión", fig_confusion(),
                        "El modelo identifica 1.413 de los 1.508 días de transporte dominante, "
                        "una exhaustividad de 0,937, a costa de 621 falsas alarmas que reducen "
                        "la precisión a 0,695. El balance responde a la ponderación equilibrada "
                        "de clases."), md=6),
                    dbc.Col(bloque("Desempeño según el horizonte", fig_horizonte(),
                        "A un día la persistencia resuelve el problema casi por completo. A "
                        "medida que el horizonte crece ambas curvas descienden, aunque la del "
                        "modelo lo hace con mayor lentitud."), md=6),
                ]),
            ),

            seccion("Sobreajuste y limitaciones",
                parrafo(
                    "La validación cruzada temporal de cinco pliegues muestra una diferencia "
                    "media de cuatro puntos porcentuales de ROC-AUC entre entrenamiento y "
                    "validación. La prueba t pareada la declara significativa (p = 0,038), "
                    "aunque su magnitud queda por debajo del rango de 5 a 10 %."),
                parrafo(
                    "El objetivo es una banda de caudales, condición no monótona que una "
                    "frontera lineal no puede representar. La variable que mide la distancia del "
                    "caudal a la descarga efectiva vuelve monótono el problema; sin ella, el "
                    "modelo lineal se estanca en un PR-AUC cercano a 0,48 en todos los "
                    "horizontes. Esa limitación de forma funcional motiva la evaluación de "
                    "modelos no lineales."),
            ),
        ),

    ], fluid=True)


# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
app.layout = dbc.Container([

    html.Div([
        html.H2("Descarga efectiva en el río Magdalena", className="mb-2"),
        html.P("Kevin Clemente Rosario y Manuel Meza Castro", className="fst-italic mb-1"),
        html.P("Universidad del Norte", className="mb-0"),
        html.P("Machine Learning - 202630", className="text-muted"),
    ], className="text-center mt-2"),
    html.Hr(),

    dcc.Tabs(id="pestanas", value="contexto", children=[
        dcc.Tab(label="Contexto del problema", value="contexto"),
        dcc.Tab(label="Análisis exploratorio", value="eda"),
        dcc.Tab(label="Modelos base", value="modelos"),
    ]),

    html.Div(id="contenido", className="mt-4"),

    html.Hr(className="mt-5"),
    html.P("Datos: IDEAM, portal DHIME.", className="text-muted small text-center"),

], fluid=True, className="p-4")


# ---------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------
@app.callback(Output("contenido", "children"), Input("pestanas", "value"))
def mostrar_pestana(pestana):
    if pestana == "contexto":
        return tab_contexto()
    elif pestana == "eda":
        return tab_eda()
    return tab_modelos()


@app.callback(Output("g-serie", "figure"), Input("sel-estaciones", "value"))
def actualizar_serie(estaciones):
    if not estaciones:
        return go.Figure()
    d = caudales.set_index("fecha")[estaciones].resample("MS").mean().reset_index()
    fig = px.line(d, x="fecha", y=estaciones,
                  labels={"value": "Caudal (m³/s)", "fecha": "Fecha",
                          "variable": "Estación"})
    fig.add_hrect(y0=LIM[0], y1=LIM[1], fillcolor="red", opacity=0.12, line_width=0)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
