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

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "NFL"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="🏈", className="header-emoji"),
                html.H1(
                    children="Team Ranking", className="header-title"
                ),
                html.P(
                    children="All the statistical ranks for each NFL team",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Type", className="menu-title"),
                        dcc.Dropdown(
                            id="team",
                            options=[
                                {"label": t, "value": t}
                                for t in np.sort(df.team.unique())
                            ],
                            value="Arizona",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Type", className="menu-title"),
                        dcc.Dropdown(
                            id="str(i)",
                            options=[
                                {"label": i, "value": i}
                                for i in df.columns
                            ],
                            value="rank_ptspergame",
                            clearable=False,
                            searchable=True,
                            className="dropdown",
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range",
                            className="menu-title"
                            ),
                        dcc.DatePickerRange(
                            id="date",
                            min_date_allowed=df.date.min().date(),
                            max_date_allowed=df.date.max().date(),
                            start_date=df.date.min().date(),
                            end_date=df.date.max().date(),
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="price-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="volume-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
    ]
)


@app.callback(
    [Output("price-chart", "figure"), Output("volume-chart", "figure")],
    [
        Input("team", "value"),
        Input("str(i)", "value"),
        Input("date", "start_date"),
        Input("date", "end_date"),
    ],
)
def update_charts(team, i, start_date, end_date):
    mask = (
        (df.team == team)
        & (df.columns == i)
        & (df.date >= start_date)
        & (df.date <= end_date)
    )
    filtered_data = df.loc[mask, :]
    price_chart_figure = {
        "data": [
            {
                "x": filtered_data["date"],
                "y": filtered_data[i],
                "type": "lines",
                "hovertemplate": "$%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Average Price of Avocados",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True},
            "yaxis": {"tickprefix": "$", "fixedrange": True},
            "colorway": ["#17B897"],
        },
    }

    volume_chart_figure = {
        "data": [
            {
                "x": filtered_data["date"],
                "y": filtered_data[i],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Avocados Sold", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }
    return price_chart_figure, volume_chart_figure


if __name__ == "__main__":
    app.run_server(debug=True)
