# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

#  Initialize Dash app
app = dash.Dash()


# Import data
df = pd.read_pickle('../data.pkl')
df['nct_id'] = df.index

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[

    html.H1(children='Hello Dash'),

    html.Div(children='Dash: A web application framework for Python.'),

    html.Label('Is study addressing cancer?'),

    dcc.RadioItems(
        id='selectcancer',
        options=[
            {'label': 'Yes', 'value': 'iscancer'},
            {'label': 'No', 'value': 'notcancer'},
        ],
        value='iscancer'
    ),

    dcc.Graph(id='graph1')
])


@app.callback(
    Output(component_id='graph1', component_property='figure'),
    [Input(component_id='selectcancer', component_property='value')])

def update_figure(selected_cancer):

    iscancer = selected_cancer=='iscancer'

    filtered_df = df[df['is_cancer']==iscancer]
    
    name = 'cancer' if iscancer else 'not cancer'

    traces =  [go.Scatter(
                    x=filtered_df['duration'],
                    y=filtered_df['droprate']*100,
                    mode='markers',
                    text=df['nct_id'],
                    opacity=0.7,
                    marker={
                        'size': 5,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=name,
                )]


    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'Study duration (months)'},
            yaxis={'title': 'Dropout rate (fraction)'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 100},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)

