#!/usr/bin/python
import argparse
import psycopg2
from config import config
from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
# import psycopg2
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
# from sklearn import datasets, linear_model
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Input parser
parser = argparse.ArgumentParser(
    description='Analyze clinical trial dropout rates.')
parser.add_argument('--plot', dest='plot', action='store_const',
                    const=True, default=False,
                    help='Create various plots (default: do not plot stuff)')
parser.add_argument('--fit', dest='fit', action='store_const',
                    const=True, default=False,
                    help='Fit linear model (default: do not fit model)')

# Custom settings
# pd.set_option('display.width', 150)
sns.set(style="white", color_codes='default', context='talk')


def connectdb():
    """ Open and return SQLAlchemy engine to PostgreSQL database """

    # read connection parameters
    params = config()

    # connect to the PostgreSQL server
    engine = create_engine('postgresql://%s:%s@%s/%s' %
                           (params['user'], params['password'],
                            params['host'], params['database']))

    return engine


def calc_dropout(engine):
    """ Given database connection, calculate enrollment & drop out totals, rate

    Args:
        engine (engine): Connection to AACT database (sqlalchemy engine
            connnection to postgresql database)

    Return:
        df (DataFrame): Table with columns for...
                        'nct_id': study ID (index)
                        'enrolled': actually study enrollment
                        'dropped': number of participants that withdrew
                        'droprate': ratio of dropped to enrolled

    Note:
    - Various filtering has been implemented on both the dropout and overall 
        study data, to ex/include in/valid data entries. See code for details
    - Use 'nct_id' as indexing column
    """
    df = None
    if engine is not None:

        # Calculate number of participants dropped from each study
        keepcols = ['count']
        renaming = {'count': 'dropped'}
        drops = pd.read_sql_table('drop_withdrawals', engine)  # withdrawl table
        filt = (
            # only look at 'Overall Study' period
            drops['period'].isin(['Overall Study']) &
            ~drops['reason'].isin(['Study Ongoing'])  # invalid dropout reason
        )
        dropdf = drops[filt].groupby('nct_id').sum(
        )[keepcols]  # accumulate total dropped
        dropdf.rename(columns=renaming, inplace=True)

        # Collect total number of participants actually enrolled in study
        keepcols = ['nct_id', 'enrollment']
        renaming = {'enrollment': 'enrolled'}
        studies = pd.read_sql_table('studies', engine)
        filt = (
            # Interventional studies only
            studies['study_type'].isin(['Interventional']) &
            # Actually enrollment numbers only
            studies['enrollment_type'].isin(['Actual']) &
            studies['overall_status'].isin(
                ['Completed'])  # Study has completed
        )
        startdf = studies[filt][keepcols]
        startdf.set_index('nct_id', inplace=True)
        startdf.rename(columns=renaming, inplace=True)
        startdf['enrolled'] = startdf['enrolled'].astype('int64')

        # Combine the tables & calculate dropout rate
        df = startdf.join(dropdf, how='inner')
        df['droprate'] = df['dropped'] / df['enrolled']

    return df


def remove_drops(df, thresh=1.0):
    """ Return dataframe with entries above a given threshold of droprate
    removed
    """
    return df[df['droprate'] < thresh]


def calc_features(engine):
    """ Given database connection, gather various study features

    Args:
        engine (engine): Connection to AACT database (sqlalchemy engine 
            connnection to postgresql database)

    Return:
        df (DataFrame): Table with columns for various features, including
            'nct_id' (study ID) as the index

    Note:
    - Use 'nct_id' as indexing column
    """

    df = pd.DataFrame()

    # ============== Data from AACT Calculated_Values table (1)
    # Table of AACT calculated values
    if engine is not None:
        keepcols = [
            'nct_id', 'registered_in_calendar_year', 'actual_duration',
            'number_of_facilities', 'has_us_facility', 'has_single_facility',
            'minimum_age_num', 'minimum_age_unit',
            'maximum_age_num', 'maximum_age_unit']
        calcvals = pd.read_sql_table('calculated_values', engine,
                                     columns=keepcols)

        # Calculate min age in years
        minimum_age_years = calcvals['minimum_age_num'].copy()
        notnull = calcvals['minimum_age_unit'].notnull()
        filt = notnull & calcvals['minimum_age_unit'].str.contains('Month')
        minimum_age_years[filt] = minimum_age_years[filt] / \
            12  # convert from months
        filt = notnull & calcvals['minimum_age_unit'].str.contains('Weeks')
        minimum_age_years[filt] = minimum_age_years[filt] / \
            52  # convert from weeks
        filt = notnull & calcvals['minimum_age_unit'].str.contains('Days')
        minimum_age_years[filt] = minimum_age_years[filt] / \
            365  # convert from days
        calcvals['minimum_age_years'] = minimum_age_years

        # Calculate max age in years
        maximum_age_years = calcvals['maximum_age_num'].copy()
        notnull = calcvals['maximum_age_unit'].notnull()
        filt = notnull & calcvals['maximum_age_unit'].str.contains('Month')
        maximum_age_years[filt] = maximum_age_years[filt] / \
            12  # convert from months
        filt = notnull & calcvals['maximum_age_unit'].str.contains('Weeks')
        maximum_age_years[filt] = maximum_age_years[filt] / \
            52  # convert from weeks
        filt = notnull & calcvals['maximum_age_unit'].str.contains('Days')
        maximum_age_years[filt] = maximum_age_years[filt] / \
            365  # convert from days
        filt = notnull & calcvals['maximum_age_unit'].str.contains('Hour')
        maximum_age_years[filt] = maximum_age_years[filt] / \
            (365 * 24)  # convert from hours
        filt = notnull & calcvals['maximum_age_unit'].str.contains('Minute')
        maximum_age_years[filt] = maximum_age_years[filt] / \
            (365 * 24 * 60)  # convert from minutes
        calcvals['maximum_age_years'] = maximum_age_years

        # Calculate age range
        calcvals['age_range_years'] = \
            calcvals['maximum_age_years'] - calcvals['minimum_age_years']

        # Select columns of interest (& rename some)
        keepcols = [
            'nct_id', 'registered_in_calendar_year', 'actual_duration',
            'number_of_facilities', 'has_us_facility', 'has_single_facility',
            'minimum_age_years', 'maximum_age_years', 'age_range_years']
        renaming = {
            'registered_in_calendar_year': 'start_year',
            'actual_duration': 'duration',
            'number_of_facilities': 'num_facilities'}
        df1 = calcvals[keepcols].copy().rename(columns=renaming)

        # Overwrite existing data
        df1.set_index('nct_id', inplace=True)
        df = df1

    # ============== Data from AACT Conditions table (2)
    if engine is not None:
        # Table of AACT-determined conditions
        conditions = pd.read_sql_table('conditions', engine,
                                       columns=['nct_id', 'downcase_name'])

        # Does this condition include the word 'cancer'?
        conditions['is_cancer'] = conditions['downcase_name'].str.contains(
            '|'.join(('cancer', '[a-z]+oma', 'leukemia', 'tumor')))

        # Collect to the study level
        df2 = conditions[['nct_id', 'is_cancer']].groupby('nct_id').any()

        # Merge with existing data
        df = df.join(df2, how='inner')

    return df


def diagnotic_plots(res, show=False):
    """ Plot diagnostic plots regression results object 
    
    Args:
        res (statsmodels.regression.linear_model.RegressionResultsWrapper):
            Results of fitting linear regression model

    Kwargs:
        show (bool): If true, call the matplotlib.pyplot.show on each figure
                     before exiting (default: False)

    Return:
        fig (tuple of matplotlib.figure.Figure): figure handles to...
            fig[0]  Historam  of fit residuals (check normality)
            fig[1]  Plot of predicted values vs residuals (check homogeneity)
    """

    # Histogram of residuals
    f1, ax1 = plt.figure(figsize=(5,4)), plt.axes()
    sns.distplot(res.resid, bins=50, kde=False)
    sns.despine(left=True)
    ax1.set(yticks=[], xlabel='droprate_tform residual')
    f1.tight_layout()

    # Plot residual vs predicted
    f2, ax2 = plt.figure(figsize=(5,4)), plt.axes()
    plt.plot(res.predict(), res.resid.values, '.')
    ax2.set(xlabel='predicted', ylabel='residual')
    f2.tight_layout()

    if show:
        f1.show()
        f2.show()

    return (f1, f2)


def get_data(savename=None):
    """ Connect to AACT database and gather data/featuresof interest
    
    Kwargs:
        savename (string): If not None, save the resulting DataFrame with
                           data to this file name using pickle
    
    Return:
        df (DataFrame): Pandas DataFrame with data features and responses
    """

    # establish connection to trials database
    engine = connectdb()

    # Collect data
    df = calc_features(engine).join(calc_dropout(engine), how='inner')

    # Remove high drop rates
    df = remove_drops(df)

    # Save
    if savename is not None:
        df.to_pickle(savename)

    # Return
    return df


if __name__ == "__main__":
    # Gather command line options
    args = parser.parse_args()

    # Gather data
    df = get_data(savename='data.pkl')

    # Plot stuff
    if args.plot:

        # Number of study participants histogram
        f, ax = plt.subplots(figsize=(5, 4))
        sns.distplot(df['enrolled'],
                     bins=np.linspace(0, 1000, num=100),
                     kde=False)
        sns.despine(left=True)
        ax.set(yticks=[], xlabel='Participants enrolled')
        f.tight_layout()
        f.show()

        # Dropout rate +/- modification
        f, ax = plt.subplots(figsize=(5, 4))
        kwargs = {'bins': np.linspace(0, 1.1, 110), 'kde': False,
                  'norm_hist': True}
        sns.distplot(df['droprate'], **kwargs, label='raw')
        sns.distplot(df['droprate_tform'], **kwargs, label='transformed')
        sns.despine(left=True)
        ax.set(yticks=[], xlabel='dropout rate (fraction)')
        ax.legend()
        f.tight_layout()
        f.show()

        # Dropout rate +/- cancer
        f, ax = plt.subplots(figsize=(5, 4))
        kwargs = {'bins': np.linspace(0, 1.1, 50),
                  'kde': False, 'norm_hist': True}
        sns.distplot(df[df['is_cancer'].notnull() & df['is_cancer'] == True]['droprate_tform'],
                     **kwargs, label='cancer')
        sns.distplot(df[df['is_cancer'].notnull() & df['is_cancer'] == False]['droprate_tform'],
                     **kwargs, label='not cancer')
        sns.despine(left=True)
        ax.set(yticks=[], xlabel='dropout rate (fraction, transformed)')
        ax.legend()
        f.tight_layout()
        f.show()

        # Study duration histogram
        f, ax = plt.subplots(figsize=(5, 4))
        sns.distplot(df[df['duration'].notnull()]['duration'],
                     bins=np.linspace(0, 200, 50), kde=False)
        sns.despine(left=True)
        ax.set(yticks=[], xlabel='Study duration (months)')
        f.tight_layout()
        f.show()

    # Implement linear model
    if args.fit:
        # Implement linear model (via statsmodels)
        formula = ('droprate**2 ~ ' +
                   'duration + has_us_facility + is_cancer')
        model = smf.ols(formula, data=df)
        res = model.fit()
        print(res.summary())

        # Check residuals for normality, homogeneity
        for x in diagnotic_plots(res):
            x.show()

        # Get predicted values & confidence intervals
        predstd, interval_l, interval_u = wls_prediction_std(res)

        # - Gather subset of data of interest
        interval_l_df = interval_l.to_frame(name='lower')
        interval_u_df = interval_u.to_frame(name='upper')
        intervals = interval_l_df.join(interval_u_df)
        model_data = df[['duration','droprate_tform','is_cancer']].\
            join(intervals, how='inner')
        model_data['pred_droprate'] = res.predict()
        model_data = model_data.sort_values('duration')

        # - Plot predicted value / CIs
        x = model_data['duration']
        y = model_data['droprate_tform']
        ypred = model_data['pred_droprate']
        ypred_l = model_data['lower']
        ypred_u = model_data['upper']

        f, ax = plt.subplots(ncols=2, figsize=(10,4))
        for cval in [False, True]:
            filt = model_data['is_cancer']==cval
            x = model_data[filt]['duration']
            y = model_data[filt]['droprate_tform']
            yp = model_data[filt]['pred_droprate']
            yl = model_data[filt]['lower']
            yu = model_data[filt]['upper'] 
            ax[int(cval)].scatter(x, y, marker='o', alpha=0.75)
            ax[int(cval)].plot(x, yp, '-', color='k')
            ax[int(cval)].fill_between(x, yl, yu, alpha=0.25, label='95%CI')
            ax[int(cval)].set(title='is_cancer {}'.format(cval),
                              xlabel='study duration',
                              ylabel='droprate_tform')
            ax[int(cval)].legend()
        f.show()