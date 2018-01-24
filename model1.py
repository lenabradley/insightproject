"""
model1 - a module to fit a linear model to clinical trials data
=========================================================================

**model1** is a python module for fitting an ordinary linear regression model,
including feature scaling and LASSO (variable selection & regularization) via
scikit-learn
"""
import pickle as pk
import pandas as pd
import data


def getmodeldata(getnew=False, **kwargs):
    """ Gather data from the 'data' module

    Args:
    getnew (bool): Default False. If True, extract data from scrach. If False, 
        load from file. If True, pass additional kwargs to data.get_data()

    Returns:
        X (dataframe): Features as a numpy array
        y (dataframe): Response
        human_names (dict): dictionary mapping columns to human-readable names
    """

    # Either gather new data or load from file
    if getnew:
        default_args = {'savename': 'rawdata.pkl',
                        'savename_human': 'human_names.pkl',
                        'N': 50,
                        'dropna': True,
                        'fill_intelligent': True}
        inputargs = {**default_args, **kwargs}
        (df, human_names) = data.get_data(**inputargs)
        [df, df_test] = data.split_data(df, save_suffix='data')

    else:
        df = pd.read_pickle('training_data.pkl')
        df_test = pd.read_pickle('testing_data.pkl')
        with open('human_names.pkl', 'rb') as input_file:
            human_names = pk.load(input_file)

    # Convert response and features to matrices
    response_names = ['dropped', 'enrolled']
    feature_names = []
    for c in df.columns.tolist():
        if c not in response_names:
            feature_names.append(c)

    X = df[feature_names]
    tmpdf = df
    tmpdf['droprate'] = tmpdf['dropped']/tmpdf['enrolled']
    y = tmpdf[['droprate']]

    return (X, y, human_names)
