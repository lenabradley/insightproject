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
currdata = res.model.data.frame.iloc[1000:1001]
pred = res.predict(currdata)

#  =================  Dash app
app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[

    html.Div(children=[
        html.H1(
            children="Let's Get Clinical!"),
        html.H5(
            children='Lena Bartell'),
        html.H3(
            children='Enter details about your trial:',
            style={'text-align':'left'}),
        ],
        id='title_block',
        style={'text-align':'center'}
    ),
    
    html.Div(children=[
        html.Label('1. What year will you start?'),
        html.Div(
            children=[],
            id='report_start_year',
            style={'text-align':'right'}),        
        dcc.Slider(
            min=1999,
            max=2020,
            step=1,
            marks={i: ' ' if i%5 else '{}'.format(i) for i in range(2000, 2021)},
            value=currdata['start_year'][0],
            id='start_year_slider')
        ],
        style={'margin-top': 40, 'margin-bottom': 10}
    ),

    html.Div(children=[
        html.Label('2. Does your study deal with any form of cancer?'),
        dcc.RadioItems(
            options=[{'label': 'Yes', 'value': True},
                     {'label': 'No', 'value': False}],
            value=currdata['is_cancer'][0],
            id='is_cancer_radio')
        ],
        id='is_cancer',
        style={'margin-top': 40, 'margin-bottom': 10}
    ),

    html.Div(children=[
        html.H3(children=[],
            id='pred_report'),
        html.Ul(children=[
            html.Li('The most influential factor is....', id='pred_bullet_1')
            ])
        ],
        style={'margin-top': 40, 'margin-bottom': 10}
    )

],
style={'align':'center', 'width':'50%', 'margin':'auto'}
)


# If start_year_slider changes, udpdate report_start_year
@app.callback(
    Output('report_start_year', 'children'),
    [Input('start_year_slider', 'value')]
    )

def update_start_year_report(input_value):
    return 'Selected: {}'.format(input_value)


# If any parameters change, update the prediction
@app.callback(
    Output(component_id='pred_report', component_property='children'),
    [Input('start_year_slider', 'value'),
     Input('is_cancer_radio', 'value')])

def update_prediction(start_year, 
                      is_cancer,
                      currdata=currdata):
    # set features
    currdata['start_year'] = start_year
    currdata['is_cancer'] = is_cancer
    
    # update prediction
    pred = res.predict(currdata)

    # Predition report string
    pred_str = 'Your predicted drop out rate is: {:d}%'.format(int(np.round(pred[0]*100)))

    return pred_str


if __name__ == '__main__':
    app.run_server(debug=True)

