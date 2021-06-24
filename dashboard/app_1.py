import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

df = pd.read_csv("data/fullranks.csv")
df.head(1)
del df["Unnamed: 0"]

df_tam = df[df["team"] == "Tampa Bay"]
df_tam["date"] = pd.to_datetime(df_tam["date"], format="%Y-%m-%d")
df_tam.sort_values("date", inplace=True)

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="NFL",),
        html.P(
            children="NFL graphs"
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": df_tam["date"],
                        "y": df_tam["rank_ptspergame"],
                        "type": "lines"
                    },
                ],
                "layout": {"title": "points per game"},
            },
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": df_tam["date"],
                        "y": df_tam["opp_rank_ptspergame"],
                        "type": "lines",
                    },
                ],
                "layout": {"title": "opp points per game"},
            },
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
