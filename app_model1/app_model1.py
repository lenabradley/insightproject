# -*- coding: utf-8 -*-
# 
import pickle as pk
import pandas as pd
# from math import ceil
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from titlecase import titlecase
# import plotly.graph_objs as go
# import numpy as np
# import base64
from json_tricks import dumps, loads

# ================== IMPORT DATA, METADATA, and MODEL

# DATA
filename = 'Xraw_model1.pkl'
with open(filename, 'rb') as input_file:
    Xraw = pk.load(input_file)

filename = 'yraw_model1.pkl'
with open(filename, 'rb') as input_file:
    yraw = pk.load(input_file)

filename = 'X_model1.pkl'
with open(filename, 'rb') as input_file:
    X = pk.load(input_file)

filename = 'y_model1.pkl'
with open(filename, 'rb') as input_file:
    y = pk.load(input_file)

with open('column_info.pkl', 'rb') as input_file:
    column_info = pk.load(input_file)
column_info['name'] = [x.capitalize() for x in column_info['name']]

# MODEL
filename = 'reg_model1.pkl'
with open(filename, 'rb') as input_file:
    reg = pk.load(input_file)

# ================= SETUP CATEGORICAL OPTIONS LIST

# (1) Conditions
filt = column_info['is_cond_']
label_names = column_info[filt]['name'].tolist()
label_vals = column_info[filt].index.tolist()
cond_options = []
for (n, v) in zip(label_names, label_vals):
    cond_options.append({'label': n, 'value': v})

# (2) Interventions
filt = column_info['is_intv_']
label_names = column_info[filt]['name'].tolist()
label_vals = column_info[filt].index.tolist()
intv_options = []
for (n, v) in zip(label_names, label_vals):
    intv_options.append({'label': n, 'value': v})

# (3) Intervention types
filt = column_info['is_intvtype_']
label_names = column_info[filt]['name'].tolist()
label_vals = column_info[filt].index.tolist()
intvtype_options = []
for (n, v) in zip(label_names, label_vals):
    intvtype_options.append({'label': n, 'value': v})

# (4) Keywords
filt = column_info['is_keyword_']
label_names = column_info[filt]['name'].tolist()
label_vals = column_info[filt].index.tolist()
keyword_options = []
for (n, v) in zip(label_names, label_vals):
    keyword_options.append({'label': n, 'value': v})

# (5) Phase
filt = column_info['is_phase']
label_names = column_info[filt]['name'].tolist()
label_vals = column_info[filt].index.tolist()
phase_options = []
for (n, v) in zip(label_names, label_vals):
    phase_options.append({'label': n, 'value': v})


# ================== SETUP DEFAULT DATA (DICT)

userdata = dict(**Xraw.iloc[0], **yraw.iloc[0])

# force start year to be 2018
userdata['year'] = 2018




#  =================  BUILD DASH APP LAYOUT
app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[

    # ==== Title (10 cols)
    html.Div(children=[
        html.H1(children="Let's Get Clinical!"),
        html.H5(children='Lena Bartell',
                style={'font-style': 'italic'})],
        id='title_block',
        style={'text-align': 'center'},
        className='twelve columns'),

    # ==== User set parameters (5 cols)
    html.Div(children=[
        html.H3('Enter details about your trial:'),

        # Number of participants you need
        html.Div(children=[
            html.Label('Number of participants needed'),
            dcc.Input(
                value='{}'.format(userdata['completed']), 
                type='text',
                id='participants_text')]),

        # Intervention Class
        html.Div(children=[
            html.Label('Intervention class (select one)'),
            dcc.Dropdown(
                options=intvtype_options,
                value=[],
                multi=False)],
            className='five columns',
            style={'width': '100%', 'margin-left': 0, 'margin-top': 20, 'margin-bottom': 20}),

        # duration
        html.Div(children=[
            html.Label('Study duration (years)',
                style={'margin-top': 20, 'margin-bottom':-20}),
            html.Div(
                children=[],
                id='report_duration',
                style={'text-align':'right'}),        
            dcc.Slider(
                min=0, max=240, step=6,
                marks={i: ' ' if i%(1*12) else '{}'.format(int(i/12)) for i in range(0, 240+1, 12)},
                value=round(userdata['duration']/12)*12,
                id='duration_slider')]),

        # num_facilities
        html.Div(children=[
            html.Label(column_info.loc['facilities', 'name'],
                style={'margin-top': 50, 'margin-bottom':-20}),
            html.Div(children=[],
                id='report_num_facilities',
                style={'text-align': 'right'}),
            dcc.Slider(
                min=0, max=250, step=1,
                marks={i: ' ' if i%50 else '{}'.format(i) for i in range(0, 250+1, 10)},
                value=userdata['facilities'],
                id='num_facilities_slider')]),

        # has_us_facility
        html.Div(children=[
            html.Label('Facilities in the US'),
            dcc.RadioItems(
                options=[{'label': 'Yes', 'value': True},
                         {'label': 'No', 'value': False}],
                value=userdata['usfacility'],
                id='has_us_facility_radio')],
            style={'margin-top': 30, 'margin-bottom': 10}),

        # Conditions
        html.Div(children=[
            html.Label('Conditions (Select all that apply)'),
            dcc.Dropdown(
                options=cond_options,
                value=[],
                multi=True)],
            className='five columns',
            style={'width': '100%', 'margin': 0, 'margin-top': 0}),

        # Interventions
        html.Div(children=[
            html.Label('Interventions (Select all that apply)'),
            dcc.Dropdown(
                options=intv_options,
                value=[],
                multi=True)],
            className='five columns',
            style={'width': '100%', 'margin': 0, 'margin-top': 20, 'margin-bottom': 20}),

        # # start_year
        # html.Div(children=[
        #     html.Label('Year trial will start',
        #         style={'margin-bottom':-20}),
        #     html.Div(children=[],
        #         id='report_start_year',
        #         style={'text-align':'right'}),        
        #     dcc.Slider(
        #         min=2000, max=2018, step=1,
        #         marks={i: ' ' if i%5 else '{}'.format(i) for i in range(2000, 2018+1)},
        #         value=userdata['year'],
        #         id='start_year_slider')]),

        ],
        className='five columns'
        ),


    # ==== Report dropout rate prediction (5 cols)
    html.Div(children=[
        html.H3('Your predictions:')
        ],
        className='five columns'),

    html.Div(children=[
        html.H3(id='pred_report'),
        html.Ul(children=[html.Li(id='pred_report_CI')]),
        html.H3('Influenctial factors:'),
        html.Ul(children=[
            html.Li(id='pred_bullet_1'),
            html.Li(id='pred_bullet_2')]),
        html.H3(id='pred_enrollment'),
        html.Ul(children=[
            html.Li('95% confident')]),
        ],
        className='five columns',
        style={'background-color':'#F89BA7', 'padding':'10px'}),


    # ==== DATA PLACEHOLDERS
    html.Div(children=[
        dumps(userdata)], 
        style={'display': 'none'},
        id='data_holder')

],
style={'margins':'auto'}
)

# ================= APP CALLBACKS


# # When userdata changes, update the prediction
# @app.callback(
#     Output('', ''),
#     [Input('data_holder', 'children')]
#     )
# def update_predictions(json_userdata):
#     return json_userdata



# If any input vars change, update the json-tricks serialized user data
@app.callback(
    Output('data_holder', 'children'),
    [Input('participants_text', 'children'),
     Input('duration_slider', 'value'),
     Input('num_facilities_slider', 'value'),
     Input('has_us_facility_radio', 'value')])
def update_userdata(participants_text, duration_slider, num_facilities_slider,
    has_us_facility_radio, userdata):

    # Modify user data
    userdata['completed'] = int(participants_text)
    userdata['duration'] = duration_slider
    userdata['facilities'] = num_facilities_slider
    userdata['usfacility'] = has_us_facility_radio

    # Return json serialzed userdata
    return dumps(userdata)





# # === If start_year_slider changes, udpdate report_start_year
# @app.callback(
#     Output('report_start_year', 'children'),
#     [Input('start_year_slider', 'value')]
#     )
# def update_report(input_value):
#     yearstr = 'Selected: {}'.format(input_value)
#     return yearstr

# === If duration_slider changes, udpdate report_duration
@app.callback(
    Output('report_duration', 'children'),
    [Input('duration_slider', 'value')]
    )
def update_report(input_value):
    outstr = 'Selected: {:0.1f}'.format(input_value/12)
    return outstr



# === If num_facilities_slider changes, udpdate report_num_facilities
@app.callback(
    Output('report_num_facilities', 'children'),
    [Input('num_facilities_slider', 'value')]
    )
def update_report(input_value):
    outstr = 'Selected: {}'.format(input_value)
    return outstr


# # If any parameters change, update the prediction
# @app.callback(
#     Output('pred_report', 'children'),
#     [Input('enrolled_text','value'),
#      Input('start_year_slider', 'value'),
#      Input('duration_slider', 'value'),
#      Input('num_facilities_slider', 'value'),
#      Input('has_us_facility_radio', 'value'),
#      Input('is_cancer_radio', 'value')])
# def update_prediction(enrolled_text, start_year, duration, num_facilities, 
#     has_us_facility, is_cancer, currdata=currdata):
#     # set features
#     currdata.iat[0, currdata.columns.get_loc('enrolled')] = int(enrolled_text)
#     currdata.iat[0, currdata.columns.get_loc('start_year')] = start_year
#     currdata.iat[0, currdata.columns.get_loc('duration')] = duration*12
#     currdata.iat[0, currdata.columns.get_loc('num_facilities')] = num_facilities
#     currdata.iat[0, currdata.columns.get_loc('has_us_facility')] = has_us_facility
#     currdata.iat[0, currdata.columns.get_loc('is_cancer')] = is_cancer

#     # update prediction
#     pred = res.get_prediction(currdata)

#     # Predition report string
#     predm = int(pred.predicted_mean[0]*100)
#     pred_str = 'Dropout rate: {:d}% '.format(predm)
#     return pred_str

# # If any parameters change, update the prediction CI
# @app.callback(
#     Output('pred_report_CI', 'children'),
#     [Input('enrolled_text','value'),
#      Input('start_year_slider', 'value'),
#      Input('duration_slider', 'value'),
#      Input('num_facilities_slider', 'value'),
#      Input('has_us_facility_radio', 'value'),
#      Input('is_cancer_radio', 'value')])
# def update_prediction(enrolled_text, start_year, duration, num_facilities, 
#     has_us_facility, is_cancer, currdata=currdata):
#     # set features
#     currdata.iat[0, currdata.columns.get_loc('enrolled')] = int(enrolled_text)
#     currdata.iat[0, currdata.columns.get_loc('start_year')] = start_year
#     currdata.iat[0, currdata.columns.get_loc('duration')] = duration*12
#     currdata.iat[0, currdata.columns.get_loc('num_facilities')] = num_facilities
#     currdata.iat[0, currdata.columns.get_loc('has_us_facility')] = has_us_facility
#     currdata.iat[0, currdata.columns.get_loc('is_cancer')] = is_cancer
    
#     # update prediction
#     pred = res.get_prediction(currdata)

#     # Predition report string
#     predl, predu = [int(x*100) for x in pred.conf_int()[0]]
#     pred_str_CI = '95% confidence interval: {:d}-{:d}%'.format(predl, predu)
#     return pred_str_CI  

# # If any parameters change, update the bullet points (#1)
# @app.callback(
#     Output(component_id='pred_bullet_1', component_property='children'),
#     [Input('start_year_slider', 'value'),
#      Input('duration_slider', 'value'),
#      Input('is_cancer_radio', 'value')])
# def update_prediction(start_year, is_cancer, currdata=currdata):
#     # factors that are (1) most significant and (2) largest effect size
#     factor_mostsig = res.pvalues.drop('Intercept').idxmin()

#     bullet_str_1 = 'The most significant factor is "{}"'.format(factor_mostsig)
#     return bullet_str_1

# # If any parameters change, update the bullet points (#2)
# @app.callback(
#     Output(component_id='pred_bullet_2', component_property='children'),
#     [Input('start_year_slider', 'value'),
#      Input('duration_slider', 'value'),
#      Input('is_cancer_radio', 'value')])
# def update_prediction(start_year, is_cancer, currdata=currdata):
#     # factors that are (1) most significant and (2) largest effect size
#     factor_biggest = res.params.drop('Intercept').idxmax()

#     bullet_str_2 = 'The dropout rate is increased the most due to "{}"'.format(factor_biggest)
#     return bullet_str_2

# # If any parameters change, update the estimated enrollment
# @app.callback(
#     Output('pred_enrollment', 'children'),
#     [Input('enrolled_text','value'),
#      Input('start_year_slider', 'value'),
#      Input('duration_slider', 'value'),
#      Input('num_facilities_slider', 'value'),
#      Input('has_us_facility_radio', 'value'),
#      Input('is_cancer_radio', 'value')])
# def update_prediction(enrolled_text, start_year, duration, num_facilities, 
#     has_us_facility, is_cancer, currdata=currdata):
#     # set features
#     currdata.iat[0, currdata.columns.get_loc('enrolled')] = int(enrolled_text)
#     currdata.iat[0, currdata.columns.get_loc('start_year')] = start_year
#     currdata.iat[0, currdata.columns.get_loc('duration')] = duration*12
#     currdata.iat[0, currdata.columns.get_loc('num_facilities')] = num_facilities
#     currdata.iat[0, currdata.columns.get_loc('has_us_facility')] = has_us_facility
#     currdata.iat[0, currdata.columns.get_loc('is_cancer')] = is_cancer

#     # update prediction
#     pred = res.get_prediction(currdata)

#     # Predited dropout rate
#     predm = pred.predicted_mean[0]
#     predl, predu = [x for x in pred.conf_int()[0]]

#     # Adjusted enrollment numbers
#     want = currdata.iat[0, currdata.columns.get_loc('enrolled')]
#     need = ceil(want / (1-predu))

#     pred_str = 'Plan to enroll {:d} participants'.format(need)
#     return pred_str


if __name__ == '__main__':
    app.run_server(debug=True)

