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
            dcc.Dropdown(
            id = "team",
            options = [{"label": i, "value": i} for i in df["team"].unique()],
            value = "value"
        ),
            dcc.Dropdown(
            id = "cols",
            options = [{"label": z, "value": z} for z in df.columns[2:]],
            value = "value"
        ),
        dcc.Graph(
            id="a_chart",
            config={"displayModeBar": False},
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
    ])

@app.callback(
    Output('a_chart', 'figure'),
    Input('team', 'value'),
    Input('cols', 'value'))

def update_charts(i, z):
    mask = ((df.team == i))
    y_val = z
    filtered_data = df.loc[mask, :]
    data_figure = {
        "data": [
            {
                "x": filtered_data["date"],
                "y": filtered_data[y_val],
                "type": "lines",
                "hovertemplate": "ranked %{y:}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "NFL team selected: " + str(i),
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True},
            "yaxis": {"tickprefix": "ranking ", "fixedrange": True},
            "colorway": ["#17B897"],
        },
    }
    return data_figure

if __name__ == "__main__":
    app.run_server(debug=True)
