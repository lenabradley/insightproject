#!/usr/bin/python
import psycopg2
from config import config
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

pd.set_option('display.width', 150)

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

    # demographic data
    base_meas = pd.read_sql_table('baseline_measurements', conn)
    
    plt.hist(base_meas['nct_id'].value_counts(), bins=np.arange(0,1000,10))
    plt.xlabel()
    plt.ylabel()
    plt.show()