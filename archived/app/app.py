# -*- coding: utf-8 -*-
from math import ceil
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import statsmodels.api as sm
import numpy as np
import base64

# ================== Import data/model
res = sm.load('../training_res.pkl')
currdata = res.model.data.frame.iloc[1000:1001]
pred = res.get_prediction(currdata)

# factors that are (1) most significant and (2) largest effect size
factor_mostsig = res.pvalues.drop('Intercept').idxmin()
factor_biggest = res.params.drop('Intercept').idxmax()


#  =================  Dash app
app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[

    # ==== Title (10 cols)
    html.Div(children=[
        # html.Img(
        #     src='https://images.unsplash.com/photo-1486825586573-7131f7991bdd?dpr=1&auto=format&fit=crop&w=1000&q=80&cs=tinysrgb', #'https://i.guim.co.uk/img/media/764988213a8826f8d1ee61c70086aad99915e198/60_0_501_590/master/501.jpg?w=300&q=55&auto=format&usm=12&fit=max&s=12bdb1dac6375cd940d661ade2a94042',
        #     style={'height':'250px'}),
        html.H1(
            children="Let's Get Clinical!"),
        html.H5(
            children='Lena Bartell',
            style={'font-style':'italic'})
        ],
        id='title_block',
        style={'text-align': 'center'},
        className='twelve columns'
    ),

    # ==== User set parameters (5 cols)
    html.Div(children=[

        html.H3('Enter details about your trial:'
        ),

        # Number of participants you need
        html.Div(children=[
            html.Label('How many participants do you need?'),
            dcc.Input(
                value='{}'.format(currdata['enrolled'][0]), 
                type='text',
                id='enrolled_text')
            ]
        ),
        
        # start_year
        html.Div(children=[
            html.Label('What year will you start?',
                style={'margin-top': 20, 'margin-bottom':-20}),
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
            ]
        ),

        # duration
        html.Div(children=[
            html.Label('How many years will the study last?',
                style={'margin-top': 50, 'margin-bottom':-20}),
            html.Div(
                children=[],
                id='report_duration',
                style={'text-align':'right'}),        
            dcc.Slider(
                min=0,
                max=20,
                step=1,
                marks={i: ' ' if i%5 else '{}'.format(i) for i in range(0, 21)},
                value=int(currdata['duration'][0]/12),
                id='duration_slider')
            ],
        ),

        # num_facilities
        html.Div(children=[
            html.Label('How many facilities will you have?',
                style={'margin-top': 50, 'margin-bottom':-20}),
            html.Div(
                children=[],
                id='report_num_facilities',
                style={'text-align':'right'}),        
            dcc.Slider(
                min=1,
                max=250,
                step=1,
                marks={i: ' ' if i%50 else '{}'.format(i) for i in range(0, 251)},
                value=currdata['num_facilities'][0],
                id='num_facilities_slider')
            ]
        ),

        # has_us_facility
        html.Div(children=[
            html.Label('Will you have facilities in the US?'),
            dcc.RadioItems(
                options=[{'label': 'Yes', 'value': True},
                         {'label': 'No', 'value': False}],
                value=currdata['has_us_facility'][0],
                id='has_us_facility_radio')
            ],
            id='has_us_facility',
            style={'margin-top': 30, 'margin-bottom': 10}
        ),

        # is_cancer
        html.Div(children=[
            html.Label('Does your study deal with any form of cancer?'),
            dcc.RadioItems(
                options=[{'label': 'Yes', 'value': True},
                         {'label': 'No', 'value': False}],
                value=currdata['is_cancer'][0],
                id='is_cancer_radio')
            ],
            id='is_cancer',
            style={'margin-top': 10, 'margin-bottom': 10}
        )

        ],
        className='five columns'
    ),

    # ==== Report dropout rate prediction (5 cols)
    html.Div(children=[
        html.H3('Your predictions:')
        ],
        className='five columns'
    ),

    html.Div(children=[
        html.H3(id='pred_report'),
        html.Ul(children=[
            html.Li(id='pred_report_CI')
            ]),        
        html.H3('Influenctial factors:'),
        html.Ul(children=[
            html.Li(id='pred_bullet_1'),
            html.Li(id='pred_bullet_2')
            ]),
        html.H3(id='pred_enrollment'),
        html.Ul(children=[
            html.Li('95% confident')
            ]),
        ],
        className='five columns',
        style={'background-color':'#F89BA7', 'padding':'10px'}
    ),
    
    # ==== Plot something cool (12 cols)
],
style={'margins':'auto'}
)


# If start_year_slider changes, udpdate report_start_year
@app.callback(
    Output('report_start_year', 'children'),
    [Input('start_year_slider', 'value')]
    )
def update_report(input_value):
    yearstr = 'Selected: {}'.format(input_value)
    return yearstr

# If duration_slider changes, udpdate report_duration
@app.callback(
    Output('report_duration', 'children'),
    [Input('duration_slider', 'value')]
    )
def update_report(input_value):
    outstr = 'Selected: {}'.format(input_value)
    return outstr


# If num_facilities_slider changes, udpdate report_num_facilities
@app.callback(
    Output('report_num_facilities', 'children'),
    [Input('num_facilities_slider', 'value')]
    )
def update_report(input_value):
    outstr = 'Selected: {}'.format(input_value)
    return outstr



# If any parameters change, update the prediction
@app.callback(
    Output('pred_report', 'children'),
    [Input('enrolled_text','value'),
     Input('start_year_slider', 'value'),
     Input('duration_slider', 'value'),
     Input('num_facilities_slider', 'value'),
     Input('has_us_facility_radio', 'value'),
     Input('is_cancer_radio', 'value')])
def update_prediction(enrolled_text, start_year, duration, num_facilities, 
    has_us_facility, is_cancer, currdata=currdata):
    # set features
    currdata.iat[0, currdata.columns.get_loc('enrolled')] = int(enrolled_text)
    currdata.iat[0, currdata.columns.get_loc('start_year')] = start_year
    currdata.iat[0, currdata.columns.get_loc('duration')] = duration*12
    currdata.iat[0, currdata.columns.get_loc('num_facilities')] = num_facilities
    currdata.iat[0, currdata.columns.get_loc('has_us_facility')] = has_us_facility
    currdata.iat[0, currdata.columns.get_loc('is_cancer')] = is_cancer

    # update prediction
    pred = res.get_prediction(currdata)

    # Predition report string
    predm = int(pred.predicted_mean[0]*100)
    pred_str = 'Dropout rate: {:d}% '.format(predm)
    return pred_str

# If any parameters change, update the prediction CI
@app.callback(
    Output('pred_report_CI', 'children'),
    [Input('enrolled_text','value'),
     Input('start_year_slider', 'value'),
     Input('duration_slider', 'value'),
     Input('num_facilities_slider', 'value'),
     Input('has_us_facility_radio', 'value'),
     Input('is_cancer_radio', 'value')])
def update_prediction(enrolled_text, start_year, duration, num_facilities, 
    has_us_facility, is_cancer, currdata=currdata):
    # set features
    currdata.iat[0, currdata.columns.get_loc('enrolled')] = int(enrolled_text)
    currdata.iat[0, currdata.columns.get_loc('start_year')] = start_year
    currdata.iat[0, currdata.columns.get_loc('duration')] = duration*12
    currdata.iat[0, currdata.columns.get_loc('num_facilities')] = num_facilities
    currdata.iat[0, currdata.columns.get_loc('has_us_facility')] = has_us_facility
    currdata.iat[0, currdata.columns.get_loc('is_cancer')] = is_cancer
    
    # update prediction
    pred = res.get_prediction(currdata)

    # Predition report string
    predl, predu = [int(x*100) for x in pred.conf_int()[0]]
    pred_str_CI = '95% confidence interval: {:d}-{:d}%'.format(predl, predu)
    return pred_str_CI  

# If any parameters change, update the bullet points (#1)
@app.callback(
    Output(component_id='pred_bullet_1', component_property='children'),
    [Input('start_year_slider', 'value'),
     Input('duration_slider', 'value'),
     Input('is_cancer_radio', 'value')])
def update_prediction(start_year, is_cancer, currdata=currdata):
    # factors that are (1) most significant and (2) largest effect size
    factor_mostsig = res.pvalues.drop('Intercept').idxmin()

    bullet_str_1 = 'The most significant factor is "{}"'.format(factor_mostsig)
    return bullet_str_1

# If any parameters change, update the bullet points (#2)
@app.callback(
    Output(component_id='pred_bullet_2', component_property='children'),
    [Input('start_year_slider', 'value'),
     Input('duration_slider', 'value'),
     Input('is_cancer_radio', 'value')])
def update_prediction(start_year, is_cancer, currdata=currdata):
    # factors that are (1) most significant and (2) largest effect size
    factor_biggest = res.params.drop('Intercept').idxmax()

    bullet_str_2 = 'The dropout rate is increased the most due to "{}"'.format(factor_biggest)
    return bullet_str_2

# If any parameters change, update the estimated enrollment
@app.callback(
    Output('pred_enrollment', 'children'),
    [Input('enrolled_text','value'),
     Input('start_year_slider', 'value'),
     Input('duration_slider', 'value'),
     Input('num_facilities_slider', 'value'),
     Input('has_us_facility_radio', 'value'),
     Input('is_cancer_radio', 'value')])
def update_prediction(enrolled_text, start_year, duration, num_facilities, 
    has_us_facility, is_cancer, currdata=currdata):
    # set features
    currdata.iat[0, currdata.columns.get_loc('enrolled')] = int(enrolled_text)
    currdata.iat[0, currdata.columns.get_loc('start_year')] = start_year
    currdata.iat[0, currdata.columns.get_loc('duration')] = duration*12
    currdata.iat[0, currdata.columns.get_loc('num_facilities')] = num_facilities
    currdata.iat[0, currdata.columns.get_loc('has_us_facility')] = has_us_facility
    currdata.iat[0, currdata.columns.get_loc('is_cancer')] = is_cancer

    # update prediction
    pred = res.get_prediction(currdata)

    # Predited dropout rate
    predm = pred.predicted_mean[0]
    predl, predu = [x for x in pred.conf_int()[0]]

    # Adjusted enrollment numbers
    want = currdata.iat[0, currdata.columns.get_loc('enrolled')]
    need = ceil(want / (1-predu))

    pred_str = 'Plan to enroll {:d} participants'.format(need)
    return pred_str


if __name__ == '__main__':
    app.run_server(debug=True)

