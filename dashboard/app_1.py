import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import pandas as pd
import numpy as np

df = pd.read_csv("data/fullranks.csv")
df.head(1)
del df["Unnamed: 0"]
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df.sort_values("date", inplace=True)

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="NFL"),
        html.P(
            children="NFL stats from teamrankings.com",
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": df["date"],
                        "y": df["rank_tdspergame"],
                        "type": "lines",
                    },
                ],
                "layout": {"title": "NFL statistical visualization"},
            },
            ),
            dcc.Dropdown(
            id = "team",
            options = [{"label": i, "value": i} for i in df["team"].unique()],
            value = "value"
        )
    ])

@app.callback(
    Output('data', 'figure'),
    Input('team', 'value'),)

def update_graph(i):
    df = df[df["team"] == i]

    return df

if __name__ == "__main__":
    app.run_server(debug=True)
