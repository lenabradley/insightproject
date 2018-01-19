# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import statsmodels.api as sm
import numpy as np
import base64

# ================== Import data/model
res = sm.load('../res.pkl')
currdata = res.model.data.frame.iloc[1000:1001]
pred = res.get_prediction(currdata)

# factors that are (1) most significant and (2) largest effect size
factor_mostsig = res.pvalues.drop('Intercept').idxmin()
factor_biggest = res.params.drop('Intercept').idxmax()

# ================= Encode image
image_filename = '/home/lena/Insight/project/reports/LoCole_Guardian.jpg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

#  =================  Dash app
app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[

    # =============================== Title (10 cols)
    html.Div(children=[
        html.Img(
            src='https://images.unsplash.com/photo-1486825586573-7131f7991bdd?dpr=1&auto=format&fit=crop&w=1000&q=80&cs=tinysrgb', #'https://i.guim.co.uk/img/media/764988213a8826f8d1ee61c70086aad99915e198/60_0_501_590/master/501.jpg?w=300&q=55&auto=format&usm=12&fit=max&s=12bdb1dac6375cd940d661ade2a94042',
            style={'height':'250px'}),
        html.H1(
            children="Let's Get Clinical!"),
        html.H5(
            children='Predicting clinical trial dropout rates',
            style={'font-style':'italic'}),
        html.H5(
            children='Lena Bartell')
        ],
        id='title_block',
        style={'text-align': 'center'},
        className='twelve columns'
    ),




    # =============================== User set parameters (5 cols)
    html.Div(children=[

        html.H3('Enter details about your trial:'
        ),
        
        # start_year
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

        # is_cancer
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
        )

        ],
        className='five columns'
    ),


    # =============================== Report dropout rate prediction (5 cols)
    html.Div(children=[

        html.H3(
            id='pred_report'
            ),
        html.Ul(children=[
            html.Li(id='pred_report_CI')
            ]),        
        html.H3('Influenctial factors:'),
        html.Ul(children=[
            html.Li(id='pred_bullet_1'),
            html.Li(id='pred_bullet_2')
            ])
        ],
        className='five columns',
        style={'background-color':'#F89BA7','padding':'10px'}
    ),

    # =============================== Plot something cool (12 cols)

],
style={'margins':'auto'}
)


# If start_year_slider changes, udpdate report_start_year
@app.callback(
    Output('report_start_year', 'children'),
    [Input('start_year_slider', 'value')]
    )
def update_start_year_report(input_value):
    yearstr = 'Selected: {}'.format(input_value)
    return yearstr

# If any parameters change, update the prediction
@app.callback(
    Output('pred_report', 'children'),
    [Input('start_year_slider', 'value'), Input('is_cancer_radio', 'value')])
def update_prediction(start_year, is_cancer, currdata=currdata):
    # set features
    currdata['start_year'] = start_year
    currdata['is_cancer'] = is_cancer
    
    # update prediction
    pred = res.get_prediction(currdata)

    # Predition report string
    predm = int(pred.predicted_mean[0]*100)
    pred_str = 'Your predicted dropout rate is: {:d}% '.format(predm)
    return pred_str

@app.callback(
    Output('pred_report_CI', 'children'),
    [Input('start_year_slider', 'value'), Input('is_cancer_radio', 'value')])
def update_prediction(start_year, is_cancer, currdata=currdata):
    # set features
    currdata['start_year'] = start_year
    currdata['is_cancer'] = is_cancer
    
    # update prediction
    pred = res.get_prediction(currdata)

    # Predition report string
    predl, predu = [int(x*100) for x in pred.conf_int()[0]]
    pred_str_CI = '95% confidence interval: {:d}-{:d}%'.format(predl, predu)
    return pred_str_CI  

# If any parameters change, update the bullet points (1)
@app.callback(
    Output(component_id='pred_bullet_1', component_property='children'),
    [Input('start_year_slider', 'value'), Input('is_cancer_radio', 'value')])
def update_prediction(start_year, is_cancer, currdata=currdata):
    # factors that are (1) most significant and (2) largest effect size
    factor_mostsig = res.pvalues.drop('Intercept').idxmin()

    bullet_str_1 = 'The most significant factor is "{}"'.format(factor_mostsig)
    return bullet_str_1

# If any parameters change, update the bullet points (2)
@app.callback(
    Output(component_id='pred_bullet_2', component_property='children'),
    [Input('start_year_slider', 'value'), Input('is_cancer_radio', 'value')])
def update_prediction(start_year, is_cancer, currdata=currdata):
    # factors that are (1) most significant and (2) largest effect size
    factor_biggest = res.params.drop('Intercept').idxmax()

    bullet_str_2 = 'Your dropout rate is increased the most due to "{}"'.format(factor_biggest)
    return bullet_str_2



if __name__ == '__main__':
    app.run_server(debug=True)

