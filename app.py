import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.graph_objects as go

def rolling_zscore(df, value_col="value", window=50):
    x = df[value_col].astype(float)
    mean = x.rolling(window, min_periods=max(5, window//5)).mean()
    std = x.rolling(window, min_periods=max(5, window//5)).std().replace(0, np.nan)
    score = ((x - mean).abs() / std).fillna(0.0)
    return score

def load_data():
    df = pd.read_csv("data/sensor_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor", "timestamp"])
    return df

df_all = load_data()
sensors = sorted(df_all["sensor"].unique().tolist())

app = Dash(__name__)
app.title = "Time-Series Anomaly Dashboard"

app.layout = html.Div([
    html.H2("Détection d’anomalies — Séries temporelles (Dash)"),

    html.Div([
        html.Label("Capteur (sensor)"),
        dcc.Dropdown(sensors, sensors[0], id="sensor_dd", clearable=False),

        html.Br(),
        html.Label("Window (rolling)"),
        dcc.Slider(20, 200, 10, value=50, id="win_slider",
                   marks={20:"20", 50:"50", 100:"100", 200:"200"}),

        html.Br(),
        html.Label("Seuil (threshold z-score)"),
        dcc.Slider(1.5, 5.0, 0.1, value=3.0, id="thr_slider",
                   marks={2:"2", 3:"3", 4:"4", 5:"5"}),
    ], style={"maxWidth":"900px"}),

    html.Br(),
    html.Div(id="kpis", style={"fontSize":"18px", "marginBottom":"10px"}),

    dcc.Graph(id="ts_graph"),

    html.H4("Top anomalies"),
    dash_table.DataTable(
        id="anomaly_table",
        columns=[
            {"name":"timestamp", "id":"timestamp"},
            {"name":"value", "id":"value"},
            {"name":"score", "id":"score"},
        ],
        page_size=10,
        style_table={"overflowX":"auto"},
        style_cell={"textAlign":"left"},
    ),
], style={"padding":"18px"})

@app.callback(
    Output("ts_graph", "figure"),
    Output("anomaly_table", "data"),
    Output("kpis", "children"),
    Input("sensor_dd", "value"),
    Input("win_slider", "value"),
    Input("thr_slider", "value"),
)
def update(sensor, window, thr):
    df = df_all[df_all["sensor"] == sensor].copy()
    df["score"] = rolling_zscore(df, "value", window=window)
    df["is_anomaly"] = df["score"] > thr

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["value"], mode="lines", name="signal"))

    anom = df[df["is_anomaly"]]
    fig.add_trace(go.Scatter(x=anom["timestamp"], y=anom["value"], mode="markers", name="anomalies"))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        margin=dict(l=40, r=20, t=30, b=40),
    )

    top = anom.sort_values("score", ascending=False).head(10)
    table_data = [
        {"timestamp": str(r["timestamp"]), "value": float(r["value"]), "score": float(r["score"])}
        for _, r in top.iterrows()
    ]

    kpis = f"Anomalies détectées: {anom.shape[0]} | Max score: {df['score'].max():.2f} | Capteur: {sensor}"
    return fig, table_data, kpis

if __name__ == "__main__":
    app.run(debug=True)

