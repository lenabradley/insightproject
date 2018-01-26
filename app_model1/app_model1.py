# -*- coding: utf-8 -*-
# 
import pickle as pk
import pandas as pd
from math import ceil
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
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
# filename = 'reg_model1.pkl'
filename = 'reg_model2.pkl'
with open(filename, 'rb') as input_file:
    reg = pk.load(input_file)

# ================= SETUP CATEGORICAL OPTIONS LIST

def get_options_list(column_info, cat_column):
    filt = column_info[cat_column]
    label_names = column_info[filt]['name'].tolist()
    label_vals = column_info[filt].index.tolist()
    options = []
    for (n, v) in zip(label_names, label_vals):
        options.append({'label': n, 'value': v})
    return options

# (1) Conditions
cond_options = get_options_list(column_info, 'is_cond_')

# (2) Interventions
intv_options = get_options_list(column_info, 'is_intv_')

# (3) Intervention types
intvtype_options = get_options_list(column_info, 'is_intvtype_')

# (4) Keywords
keyword_options = get_options_list(column_info, 'is_keyword_')

# (5) Phase
phase_options = get_options_list(column_info, 'is_phase')


# ================== SETUP DEFAULT DATA (DICT)

userdata = dict(**Xraw.iloc[0], **yraw.iloc[0])

# set userdata to initial values

# - set all categoricals to False
cat_names = column_info.loc[column_info['categorical']].index.tolist()
for n in cat_names:
    userdata[n] = False

# - set as Phase 1 & Drug
userdata['phase1'] = True
userdata['intvtype_drug'] = True

# - set continuous/numerics
userdata['year'] = 2018 # force start year to be 2018
userdata['malefraction'] = 0.5 # - 50/50 male/female
userdata['arms'] = 1 # - 1 arm
userdata['duration'] = 12*5 # 5 year study
userdata['minage'] = 18 # minimum age 18

# ================== GET CURRENT CATEGORICAL CHOICES/VALUE SETS
def get_value_list(userdata, column_info, cat_column):
    colnames = column_info[column_info[cat_column]].index.tolist()
    values = []
    for c in colnames:
        if userdata[c]:
            values.append(c)
    return values

# (1) Conditions
cond_values = get_value_list(userdata, column_info, 'is_cond_')

# (2) Interventions
intv_values = get_value_list(userdata, column_info, 'is_intv_')

# (3) Intervention types
intvtype_value = get_value_list(userdata, column_info, 'is_intvtype_')[0]

# (4) Keywords
keyword_values = get_value_list(userdata, column_info, 'is_keyword_')

# (5) Phase
phase_values = get_value_list(userdata, column_info, 'is_phase')


#  =================  BUILD DASH APP LAYOUT
app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[

    # ==== Title (10 cols)
    html.Div(children=[
        html.H1(children="Let's Get Clinical!", style={}),
        html.Hr()],
        id='title_block',
        style={'text-align': 'center'},
        className='twelve columns'),

    # ==== User set parameters (5 cols)
    html.Div(children=[
        html.H3('Enter details about your trial',
            style={'text-align': 'center'}),

        # Number of participants you need
        html.Div(children=[
            html.Label('Number of participants needed'),
            dcc.Input(
                value='{}'.format(userdata['completed']), 
                type='text',
                id='participants_text')]),

        # Fraction M/F
        html.Div(children=[
            html.Label('Percent of participants that are male'),
            dcc.Input(
                value='{}'.format(int(userdata['malefraction']*100)),
                type='text',
                id='malefraction_text')]),

        # Minimum age
        html.Div(children=[
            html.Label('Minimum age of participants (years)'),
            dcc.Input(
                value='{}'.format(userdata['minage']),
                type='text',
                id='minage_text')]),        

        # Phase
        html.Div(children=[
            html.Label('Phase'),
            dcc.Checklist(
                options=phase_options,
                values=phase_values,
                id='phase_checklist')]),

        # Intervention Class
        html.Div(children=[
            html.Label('Intervention class (select one)'),
            dcc.Dropdown(
                options=intvtype_options,
                searchable=False,
                value=intvtype_value,
                multi=False,
                id='intvtype_dropdown')],
            className='five columns',
            style={'width': '100%', 'margin-left': 0, 'margin-top': 20, 'margin-bottom': 20}),

        # number of arms
        html.Div(children=[
            html.Label('Number of study arms',
                style={'margin-top': 20, 'margin-bottom':-20}),
            html.Div(
                children=[],
                id='report_arms',
                style={'text-align':'right'}),        
            dcc.Slider(
                min=0, max=20, step=1,
                marks={i: ' ' if i%5 else '{}'.format(int(i)) for i in range(0, 20+1)},
                value=userdata['arms'],
                id='arms_slider')]),

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
                placeholder='Search for terms...',
                value=cond_values,
                multi=True,
                id='cond_dropdown')],
            className='five columns',
            style={'width': '100%', 'margin': 0, 'margin-top': 0}),

        # Interventions
        html.Div(children=[
            html.Label('Interventions (Select all that apply)'),
            dcc.Dropdown(
                options=intv_options,
                placeholder='Search for terms...',
                value=intv_values,
                multi=True,
                id='intv_dropdown')],
            className='five columns',
            style={'width': '100%', 'margin': 0, 'margin-top': 20}),

        # Keywords
        html.Div(children=[
            html.Label('Keywords (Select all that apply)'),
            dcc.Dropdown(
                options=keyword_options,
                placeholder='Search for terms...',
                value=keyword_values,
                multi=True,
                id='keyword_dropdown')],
            className='five columns',
            style={'width': '100%', 'margin': 0, 'margin-top': 20, 'margin-bottom': 20}),        

        ],
        className='six columns',
        style={'background-color':'LightGray', 'padding': '20px'}
        ),


    # ==== Report dropout rate prediction (5 cols)
    # Predictions title
    html.Div(children=[
        html.H3('Your predictions', style={'text-align': 'center'}),
        html.Div(children=[html.H5()],
            id='pred_report')
        ],
        className='five columns',
        style={'background-color':'#fdc086', 'padding': '20px'}),

    # # Predictions box
    # html.Div(children=[
    #     html.H5(),
    #     html.H5()],
    #     id='pred_report',
    #     className='five columns',
    #     style={'background-color':'#fed9a6', 'padding': '10px'}),


    # ==== Footer
    html.Div(children=[
            html.Hr(),
            html.H5(children='Lena Bartell',
                style={'font-style': 'italic'}),
            html.H1(' ')],
        className='twelve columns',
        style={'text-align': 'center', 'padding': '50'}),


    # ==== DATA PLACEHOLDERS
    html.Div(children=[
        dumps(userdata)], 
        style={'display': 'none'},
        id='data_holder')

],
style={'margins': 'auto'}
)

# ================= APP CALLBACKS


# When userdata changes, update the prediction
@app.callback(
    Output('pred_report', 'children'),
    [Input('data_holder', 'children')]
    )
def update_predictions(json_userdata, reg=reg, Xraw=Xraw):

    # De-serialize data
    userdata = dict(loads(json_userdata))
    del userdata['droprate']

    # Format data to array for model input
    newXraw = Xraw.iloc[:0]
    newXraw = newXraw.append(userdata, ignore_index=True)
    newX = newXraw.as_matrix()

    # Predict dropout rate & create associated string
    pred_droprate = reg.predict(newX)[0]
    pred_enroll = userdata['completed'] / (1-pred_droprate)
    pred_droprate_str = ('Your predicted dropout rate is {}%, so you should ' + 
        'plan to enroll {:d} participants ').format(
        int(round(pred_droprate*100)),
        ceil(pred_enroll)
        )

    # Format callback output/children
    children = [html.H5(pred_droprate_str)]

    return children

# If any input vars change, update the json-tricks serialized user data
@app.callback(
    Output('data_holder', 'children'),
    [Input('participants_text', 'value'),
     Input('malefraction_text', 'value'),
     Input('minage_text', 'value'),
     Input('duration_slider', 'value'),
     Input('arms_slider', 'value'),
     Input('num_facilities_slider', 'value'),
     Input('has_us_facility_radio', 'value'),
     Input('intvtype_dropdown', 'value'),
     Input('cond_dropdown', 'value'),
     Input('intv_dropdown', 'value'),
     Input('phase_checklist', 'values'),
     Input('keyword_dropdown', 'value')])
def update_userdata(participants_text, malefraction_text, minage_text,
    duration_slider, arms_slider, num_facilities_slider,
    has_us_facility_radio, intvtype_dropdown, cond_dropdown, intv_dropdown,
    phase_checklist, keyword_dropdown,
    userdata=userdata, column_info=column_info):

    # Modify user data
    userdata['completed'] = int(participants_text)
    userdata['malefraction'] = int(malefraction_text)/100.
    userdata['minage'] = float(minage_text)
    userdata['duration'] = duration_slider
    userdata['arms'] = int(arms_slider)
    userdata['facilities'] = num_facilities_slider
    userdata['usfacility'] = has_us_facility_radio

    # Set intervention class options as T/F according to selected
    intvtype_names = column_info[column_info['is_intvtype_']].index.tolist()
    for n in intvtype_names:
        userdata[n] = False
        if intvtype_dropdown is not None:
            if n in intvtype_dropdown:
                userdata[n] = True

    # Set condition MeSh terms options as T/F according to selected
    cond_names = column_info[column_info['is_cond_']].index.tolist()
    for n in cond_names:
        userdata[n] = False
        if cond_dropdown is not None:
            if n in cond_dropdown:
                userdata[n] = True

    # Set intervention MeSh terms options as T/F according to selected
    intv_names = column_info[column_info['is_intv_']].index.tolist()
    for n in intv_names:
        userdata[n] = False
        if intv_dropdown is not None:
            if n in intv_dropdown:
                userdata[n] = True

    # Set phase as T/F according to selected
    phase_names = column_info[column_info['is_phase']].index.tolist()
    for n in phase_names:
        userdata[n] = False
        if phase_checklist is not None:
            if n in phase_checklist:
                userdata[n] = True

    # Set keywords T/F according to selected
    keyword_names = column_info[column_info['is_keyword_']].index.tolist()
    for n in keyword_names:
        userdata[n] = False
        if keyword_dropdown is not None:
            if n in keyword_dropdown:
                userdata[n] = True                

    # Return json serialzed userdata
    return dumps(userdata)


# If duration_slider changes, udpdate report_duration
@app.callback(
    Output('report_duration', 'children'),
    [Input('duration_slider', 'value')]
    )
def update_report(input_value):
    outstr = 'Selected: {:0.1f}'.format(input_value/12)
    return outstr


# If arms_slider changes, udpdate report_duration
@app.callback(
    Output('report_arms', 'children'),
    [Input('arms_slider', 'value')]
    )
def update_report(input_value):
    outstr = 'Selected: {}'.format(int(input_value))
    return outstr    

# If num_facilities_slider changes, udpdate report_num_facilities
@app.callback(
    Output('report_num_facilities', 'children'),
    [Input('num_facilities_slider', 'value')]
    )
def update_report(input_value):
    outstr = 'Selected: {}'.format(input_value)
    return outstr


# MAIN
if __name__ == '__main__':
    app.run_server(debug=True)

