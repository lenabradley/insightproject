#!/usr/bin/python
import psycopg2
from config import config
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
# pd.set_option('display.width', 150)

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

def calc_features(conn):
    """ Given database connection, gather various study features 

    Args:
        conn (engine): Connection to AACT database (sqlalchemy engine 
            connnection to postgresql database)

    Return:
        df (DataFrame): Table with columns for various features, including
            'nct_id' (study ID) as the index

    """
    # Table of AACT calculated values
    calcvals = pd.read_sql_table('calculated_values', conn)

    # Calculate min age in years
    minimum_age_years = calcvals['minimum_age_num'].copy()
    filt = calcvals['minimum_age_unit'].isin(['Months', 'Month']) 
    minimum_age_years[filt] = minimum_age_years[filt]/12 # convert from months
    filt = calcvals['minimum_age_unit'].isin(['Weeks'])
    minimum_age_years[filt] = minimum_age_years[filt]/52 # convert from weeks
    filt = calcvals['minimum_age_unit'].isin(['Days'])
    minimum_age_years[filt] = minimum_age_years[filt]/365 # convert from days
    calcvals['minimum_age_years'] = minimum_age_years

    # Calculate max age in years
    maximum_age_years = calcvals['maximum_age_num'].copy() 
    filt = calcvals['maximum_age_unit'].isin(['Months', 'Month']) 
    maximum_age_years[filt] = maximum_age_years[filt]/12 # convert from months
    filt = calcvals['maximum_age_unit'].isin(['Weeks'])
    maximum_age_years[filt] = maximum_age_years[filt]/52 # convert from weeks
    filt = calcvals['maximum_age_unit'].isin(['Days']) 
    maximum_age_years[filt] = maximum_age_years[filt]/365 # convert from days
    calcvals['maximum_age_years'] = maximum_age_years

    # Select columns of interest (& rename some)
    keepcols = ['nct_id', 'registered_in_calendar_year', 'number_of_facilities',
        'actual_duration', 'has_us_facility', 'has_single_facility',
        'minimum_age_years', 'maximum_age_years']
    renaming = {
        'registered_in_calendar_year': 'start_year',
        'actual_duration': 'duration',
        'number_of_facilities': 'num_facilities' }
    df = calcvals[keepcols].rename(columns=renaming)
    df['age_range_years'] = df['maximum_age_years']-df['minimum_age_years']

    # Set study ID as index
    df.set_index('nct_id', inplace=True)

    return df

if __name__ == "__main__":

    # establish connection to trials database
    conn = connectdb()
    
    # # histogram of number of study participants
    # base_counts = pd.read_sql_table('baseline_counts', conn)
    # plt.hist(base_counts['count'], bins=np.arange(0,1000,50))
    # plt.xlabel('Number of study participants (people, elbows, ...)')
    # plt.ylabel('Frequency')
    # plt.title('%d studies total'%(len(base_counts['count'])))
    # plt.show()

    # # Top reasons for patient withdrawl
    # drops = pd.read_sql_table('drop_withdrawals', conn)
    # top_reasons = drops['reason'].value_counts();
    # print(top_reasons[:10])


    # # Top reasons for study termination
    # studies = pd.read_sql_table('studies', conn)
    # for (reason, number) in studies['why_stopped'].str.lower().value_counts()[:20].iteritems():
    #     print('%s\t%d'%(reason, number))

    # # Histogram of enrollment
    # studies['enrollment'].hist(bins=np.arange(0,1e3,50))
    # plt.show()

    # Collect data
    df1 = calc_dropout(conn)
    df2 = calc_features(conn)
    df = df1.join(df2, how='inner')
    df =  df[df['droprate']<1.0].copy()

    # is this treating cancer?
    df3 = pd.read_sql_table('conditions', conn)[['nct_id','downcase_name']].copy()
    df3.set_index('nct_id', inplace=True)
    df3['is_cancer'] = df3['downcase_name'].str.contains('cancer')
    df = df.join(df3, how='inner')

    g = [
        df[df['is_cancer'].notnull() & df['is_cancer']]['droprate'].values,
        df[df['is_cancer'].notnull() & (df['is_cancer']==False)]['droprate'].values
    ]
    [print('{0:.5f} +/- {1:.5f}'.format(x.mean(), x.std()/len(x))) for x in g];
































