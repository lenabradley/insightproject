# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import statsmodels.api as sm
import numpy as np

# ================== Import data/model
res = sm.load('../res.pkl')
currdata = res.model.data.frame.iloc[1000]

#  =================  Dash app
app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[

    html.H1(children='Hello Dash'),

    html.Div(children='Dash: A web application framework for Python.'),

    html.Div(children=[
        html.Label('What year will you start?'),
        dcc.Slider(
            min=1999,
            max=2020,
            step=1,
            marks={i: '{}'.format(i) for i in range(2000, 2021, 5)},
            value=2018,
        )],
        id='start_year_slider', 
        style={'margin-top': 40, 'margin-bottom': 10}
    ),

    html.Div(children=[
        html.Label('Is study addressing cancer?'),
        dcc.RadioItems(
            id='selectcancer',
            options=[
                {'label': 'Yes', 'value': 'iscancer'},
                {'label': 'No', 'value': 'notcancer'},
            ],
            value='iscancer'
        )],
        id='cancer_radio',
        style={'margin-top': 40, 'margin-bottom': 10}
    ),

    dcc.Graph(id='graph1'),
],
style={'align':'center', 'width':'50%', 'margin':'auto'})


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

