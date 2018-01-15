#!/usr/bin/python
import argparse
import psycopg2
from config import config
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn as sns

# Input parser
parser = argparse.ArgumentParser(
    description='Analyze clinical trial dropout rates.')
parser.add_argument('--plot', dest='plot', action='store_const',
    const=True, default=False,
    help='Create various plots (default: do not plot stuff)')

# Custom settings
# pd.set_option('display.width', 150)
sns.set(style="white", color_codes='default', context='talk')

def connectdb():
    """ Open and return SQLAlchemy engine connection to PostgreSQL database """

    engine = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        engine = create_engine('postgresql://%s:%s@%s/%s'%(
            params['user'], params['password'], 
            params['host'], params['database']))
    
    except:
            pass

    return engine

def calc_dropout(conn):
    """ Given database connection, calculate enrollment & drop out totals, rate

    Args:
        conn (engine): Connection to AACT database (sqlalchemy engine 
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
    if conn is not None:

        # Calculate number of participants dropped from each study 
        keepcols = ['count']
        renaming = {'count': 'dropped'}
        drops = pd.read_sql_table('drop_withdrawals', conn) # withdrawl table
        filt = (
            drops['period'].isin(['Overall Study']) & # only look at 'Overall Study' period
            ~drops['reason'].isin(['Study Ongoing']) # invalid dropout reason
            )
        dropdf = drops[filt].groupby('nct_id').sum()[keepcols] # accumulate total dropped
        dropdf.rename(columns=renaming, inplace=True)

        # Collect total number of participants actually enrolled in study
        keepcols = ['nct_id', 'enrollment']
        renaming = {'enrollment': 'enrolled'}
        studies = pd.read_sql_table('studies', conn)
        filt = (
            studies['study_type'].isin(['Interventional'])  & # Interventional studies only
            studies['enrollment_type'].isin(['Actual']) & # Actually enrollment numbers only
            studies['overall_status'].isin(['Completed']) # Study has completed
            )
        startdf = studies[filt][keepcols]
        startdf.set_index('nct_id', inplace=True)
        startdf.rename(columns=renaming, inplace=True)
        startdf['enrolled'] = startdf['enrolled'].astype('int64')

        # Combine the tables & calculate dropout rate
        df = startdf.join(dropdf, how='inner')
        df['droprate'] = df['dropped']/df['enrolled']

    return df

def remove_drops(df, thresh=1.0):
    """ Return dataframe with entries above a given threshold of droprate removed"""
    return df[df['droprate']<thresh]

def calc_features(conn):
    """ Given database connection, gather various study features 

    Args:
        conn (engine): Connection to AACT database (sqlalchemy engine 
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
    if conn is not None:
        keepcols = [
            'nct_id', 'registered_in_calendar_year', 'actual_duration',
            'number_of_facilities', 'has_us_facility', 'has_single_facility',
            'minimum_age_num', 'minimum_age_unit', 
            'maximum_age_num', 'maximum_age_unit']
        calcvals = pd.read_sql_table('calculated_values', conn,
            columns=keepcols)

        # Calculate min age in years
        minimum_age_years = calcvals['minimum_age_num'].copy()
        notnull = calcvals['minimum_age_unit'].notnull()
        filt = notnull & calcvals['minimum_age_unit'].str.contains('Month')
        minimum_age_years[filt] = minimum_age_years[filt]/12 # convert from months
        filt = notnull & calcvals['minimum_age_unit'].str.contains('Weeks')
        minimum_age_years[filt] = minimum_age_years[filt]/52 # convert from weeks
        filt = notnull & calcvals['minimum_age_unit'].str.contains('Days')
        minimum_age_years[filt] = minimum_age_years[filt]/365 # convert from days
        calcvals['minimum_age_years'] = minimum_age_years

        # Calculate max age in years
        maximum_age_years = calcvals['maximum_age_num'].copy()
        notnull = calcvals['maximum_age_unit'].notnull()
        filt = notnull & calcvals['maximum_age_unit'].str.contains('Month')
        maximum_age_years[filt] = maximum_age_years[filt]/12 # convert from months
        filt = notnull & calcvals['maximum_age_unit'].str.contains('Weeks')
        maximum_age_years[filt] = maximum_age_years[filt]/52 # convert from weeks
        filt = notnull & calcvals['maximum_age_unit'].str.contains('Days') 
        maximum_age_years[filt] = maximum_age_years[filt]/365 # convert from days
        filt = notnull & calcvals['maximum_age_unit'].str.contains('Hour') 
        maximum_age_years[filt] = maximum_age_years[filt]/(365*24) # convert from hours
        filt = notnull & calcvals['maximum_age_unit'].str.contains('Minute') 
        maximum_age_years[filt] = maximum_age_years[filt]/(365*24*60) # convert from minutes
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
            'number_of_facilities': 'num_facilities' }
        df1 = calcvals[keepcols].copy().rename(columns=renaming)

        # Overwrite existing data
        df1.set_index('nct_id', inplace=True)
        df = df1

    # ============== Data from AACT Conditions table (2)
    if conn is not None:
        # Table of AACT-determined conditions
        conditions = pd.read_sql_table('conditions', conn, 
            columns=['nct_id', 'downcase_name'])    

        # Does this condition include the word 'cancer'?
        conditions['is_cancer'] = conditions['downcase_name'].str.contains(
            '|'.join(('cancer', '[a-z]+oma', 'leukemia', 'tumor')))

        # Collect to the study level
        df2 = conditions[['nct_id','is_cancer']].groupby('nct_id').any()

        # Merge with existing data
        df = df.join(df2, how='inner')

    return df

def tform_var(x, pow=0.5, plothist=False):
    """ Transform input array values by the given power 

    Args:
        x (numpy.ndarray or similar): array to transform
    Kwargs:
        pow (float): Power apply to transform array
        plothist (bool): Plot histogram of transformed array (default: False)

    Return:
        xtform (same as x): Transformed array
    """
    xtform = None
    try:
        xtform = x**pow
    except:
        pass

    if xtform is not None and plothist:
        plt.figure()
        plt.hist(xtform, bins=50)
        plt.show();

    return xtform

if __name__ == "__main__":
    # Gather command line options
    args = parser.parse_args()

    # establish connection to trials database
    conn = connectdb()

    # Collect data
    df = calc_features(conn).join(calc_dropout(conn), how='inner')

    # Transform droprate to be more normally distributed
    df['droprate_tform'] = tform_var(df['droprate'], pow=0.4)

    # # Remove high drop rates
    # df = remove_drops(df)

    # Plot stuff
    if args.plot:

        # Number of study participants histogram
        f, ax = plt.subplots(figsize=(5, 4))
        sns.distplot(df['enrolled'], bins=np.linspace(0, 1000, num=100), kde=False)
        sns.despine(left=True)
        ax.set(yticks=[], xlabel='Participants enrolled')
        f.tight_layout()
        f.show()

        # Dropout rate +/- modification
        f, ax = plt.subplots(figsize=(5, 4))
        kwargs = {'bins':np.linspace(0, 1.1, 110), 'kde':False, 'norm_hist':True}
        sns.distplot(df['droprate'], **kwargs, label='raw')
        sns.distplot(df['droprate_tform'], **kwargs, label='transformed')
        sns.despine(left=True)
        ax.set(yticks=[], xlabel='dropout rate (fraction)')
        ax.legend()
        f.tight_layout()
        f.show()

        # Dropout rate +/- cancer
        f, ax = plt.subplots(figsize=(5, 4))
        kwargs = {'bins':np.linspace(0, 1.1, 50), 'kde':False, 'norm_hist':True}
        sns.distplot(df[df['is_cancer'].notnull() & df['is_cancer']==True]['droprate_tform'], 
            **kwargs, label='cancer')
        sns.distplot(df[df['is_cancer'].notnull() & df['is_cancer']==False]['droprate_tform'], 
            **kwargs, label='not cancer')
        sns.despine(left=True)
        ax.set(yticks=[], xlabel='dropout rate (fraction, transformed)')
        ax.legend()
        f.tight_layout()
        f.show()

        # Study duration histogram
        f, ax = plt.subplots(figsize=(5,4))
        sns.distplot(df[df['duration'].notnull()]['duration'], 
            bins=np.linspace(0,200,50), kde=False)
        sns.despine(left=True)
        ax.set(yticks=[], xlabel='Study duration (months)')
        f.tight_layout()
        f.show()


        # Exploratory plots
        # f, ax = plt.subplots(figsize=(5,4))













