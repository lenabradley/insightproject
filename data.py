"""
data - a module to extract and process AACT clinical trials data
=================================================================

**data** is a python module for connecting to the AACT relational database
(via PostgreSQL), extract features of interest, and clean/process that data
"""

from sqlalchemy import create_engine, MetaData, func, or_, and_, distinct
from sqlalchemy.orm import sessionmaker
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import re
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle as pk
from configparser import ConfigParser
from nltk.tokenize import RegexpTokenizer


def _config(filename='database.ini', section='postgresql'):
    """ Configure parameters from specified section """

    # create a parser
    parser = ConfigParser()

    # read config file
    parser.read(filename)
 
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
 
    return db


def _connectdb():
    """ Open and return SQLAlchemy engine to PostgreSQL database """

    # read connection parameters
    params = _config()

    # connect to the PostgreSQL server
    conn = create_engine('postgresql://%s:%s@%s/%s' %
                           (params['user'], params['password'],
                            params['host'], params['database']))

    meta = MetaData(bind=conn)
    meta.reflect()

    return conn, meta


def list_group(df, group_column=None, value_column=None):
    """ Given dataframe, use numpy to group rows by group_column and return a 
    dataframe with all corresponding values in value_column as a list

    Kwargs:
        df (dataframe): Dataframe with columns to group_by and list
        group_column (string): Name of column to group on
        value_column (string): Name of column for which to return all grouped 
            values as a list

    Returns:
        df2 (dataframe): A new data frame with two columns (group_column and 
            value_column), where value_column is a list of the corresponding 
            group entries

    Notes:
        - The resulting dataframe only contains the two parameter columns, all 
        other information in the input dataframe is lost
    """
    keys, values = df.sort_values(group_column).values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])
    df2 = pd.DataFrame({group_column: ukeys,
                        value_column: [list(a) for a in arrays]}) 
    return df2[[group_column, value_column]]


def _gather_response(conn=None, meta=None):
    """ Connect to AACT postgres database and collect response variables (number
    of participants started, completed, not completed)

    Kwargs:
        conn (engine): Sqlalchemy engine connection to database. If None, a new 
            connection will be established. Default is None
        meta (metadata): Sqlalchemy metadata scheme from conn. If None, it will
            be gathered from a new connection

    Returns:
        df (DataFrame): Pandas dataframe with columns for study ID ('nct_id'), 
            number of participants enrolled ('enrolled') at the start, and the
            number that dropped out ('dropped')
        human_names (dict): dictionary mapping columns to human-readable names
    """

    # Initialize output dataframe
    df = None

    # Connect to database, if necessary, & open session
    if conn is None or meta is None:
        conn, meta = _connectdb()
    Session = sessionmaker(bind=conn)
    session = Session()

    # Milestone counts of started / completed / not completed
    table = meta.tables['milestones']
    for title in ['STARTED', 'COMPLETED', 'NOT COMPLETED']:
        # set new column name as lowercase, no spaces
        newtitle = title.lower().replace(' ', '')

        # setup & run sql query
        grp = table.c.nct_id
        cols = (grp, func.sum(table.c.count).label(newtitle))
        filt = table.c.title == title
        query = session.query(*cols).filter(filt).group_by(grp)

        # Gather results and merge with existing dataframe, if applicable
        newdf = pd.read_sql(query.statement, query.session.bind)
        if df is None:
            df = newdf
        else:
            df = df.merge(newdf, on='nct_id', how='outer')

    # Calculate dropout rate (if invalid, set to -1)
    df['droprate'] = df['notcompleted'] / df['started']
    df['droprate'] = df['droprate'].fillna(-1)
    df['droprate'] = df['droprate']

    # human-readable names
    human_names = {'started': 'number of participants',
                   'completed': 'number participants completed',
                   'notcompleted': 'number of participants dropped',
                   'droprate': 'dropout rate (fraction)'}
    name_check = human_names.keys()
    for k in name_check:
        if k not in df:
            del human_names[k]

    return df, human_names


def _gather_sex(conn=None, meta=None):
    """ Connect to AACT database and calculate study participant counts by sex

    Kwargs:
        conn (engine): Sqlalchemy engine connection to database. If None, a new 
            connection will be established. Default is None
        meta (metadata): Sqlalchemy metadata scheme from conn. If None, it will
            be gathered from a new connection

    Returns:
        df (dataframe): Pandas dataframe with study id ('nct_id') and total 
            number of participants, separated by sex ('male', and 'female')
        human_names (dict): dictionary mapping columns to human-readable names
    """

    # Connect to database, if necessary, & open session
    if conn is None or meta is None:
        conn, meta = _connectdb()
    Session = sessionmaker(bind=conn)
    session = Session()    

    # Initialize output dataframe
    df = None

    # prep for query
    table = meta.tables['baseline_measurements']
    grp_cols = (table.c.category, table.c.classification)

    # Group counts by sex (fe/male)
    sexes = ['male', 'female']
    for sex in sexes:
        cols = (table.c.nct_id, func.sum(table.c.param_value_num).label(sex))
        filt = or_(c.ilike(sex) for c in grp_cols)
        query = session.query(*cols).filter(filt).group_by(table.c.nct_id)

        # Merge with existing dataframe, if applicable
        newdf = pd.read_sql(query.statement, query.session.bind)
        if df is None:
            df = newdf
        else:
            df = df.merge(newdf, on='nct_id', how='outer')

    # if no data, assume group size of zero
    for sex in sexes:
        df[sex] = df[sex].fillna(0).astype(int)

    # calculate fraction male
    df['malefraction'] = df['male'] / (df['male'] + df['female'])
    df = df[['nct_id', 'malefraction']]

    # human-readable names
    human_names = {'malefraction': 'fraction male'}
    name_check = human_names.keys()
    for k in name_check:
        if k not in df:
            del human_names[k]

    return df, human_names


def _gather_conditions(conn=None, meta=None, N=10):
    """ Connect to AACT database and gather list of top N most common MESH terms
    of conditions, group by study ID (nct_id)

    Args:
        conn (engine): Sqlalchemy engine connection to database. If None, a new 
            connection will be established. Default is None
        meta (metadata): Sqlalchemy metadata scheme from conn. If None, it will
            be gathered from a new connection    
        N (int): Only keep the top N most common terms/keywords. Default 10

    Return:
        df (DataFrame): pandas dataframe with columns for study ID ('nct_id'), 
            and list of top N condition mesh terms ('conditions')
        human_names (dict): dictionary mapping columns to human-readable names    
    """
    # Connect to database, if necessary, & open session
    if conn is None or meta is None:
        conn, meta = _connectdb()
    Session = sessionmaker(bind=conn)
    session = Session()

    # Initialize output dataframe
    df = None

    # prep for query
    table = meta.tables['browse_conditions']

    # Gather conditions vocabulary (top N most frequent terms)
    col = table.c.downcase_mesh_term.label('term')
    countcol = func.count(col).label('count')
    query = session.query(col).group_by(col).order_by(countcol.desc()).limit(N)
    top_terms = [x[0] for x in query.all()]

    # Collect list of top terms in each study
    cols = (table.c.nct_id, table.c.downcase_mesh_term.label('conditions'))
    filt = or_(table.c.downcase_mesh_term == term for term in top_terms)
    query = session.query(*cols).filter(filt)
    newdf = pd.read_sql(query.statement, query.session.bind)
    df = list_group(newdf, group_column='nct_id', value_column='conditions')

    # Human-readable names
    human_names = {'conditions': 'list of condition MESH terms'}

    return df, human_names


def _gather_interventions(conn=None, meta=None, N=10):
    """ Connect to AACT database and gather list of top N most common MESH terms
    of interventions, group by study ID (nct_id)

    Args:
        conn (engine): Sqlalchemy engine connection to database. If None, a new 
            connection will be established. Default is None
        meta (metadata): Sqlalchemy metadata scheme from conn. If None, it will
            be gathered from a new connection    
        N (int): Only keep the top N most common terms/keywords. Default 10

    Return:
        df (DataFrame): pandas dataframe with columns for study ID ('nct_id'), 
            and list of top N intervention mesh terms ('interventions')
        human_names (dict): dictionary mapping columns to human-readable names    
    """
    # Connect to database, if necessary, & open session
    if conn is None or meta is None:
        conn, meta = _connectdb()
    Session = sessionmaker(bind=conn)
    session = Session()

    # Initialize output dataframe
    df = None

    # prep for query
    table = meta.tables['browse_interventions']

    # Gather conditions vocabulary (top N most frequent terms)
    col = table.c.downcase_mesh_term.label('term')
    countcol = func.count(col).label('count')
    query = session.query(col).group_by(col).order_by(countcol.desc()).limit(N)
    top_terms = [x[0] for x in query.all()]

    # Collect list of top terms in each study
    cols = (table.c.nct_id, table.c.downcase_mesh_term.label('interventions'))
    filt = or_(table.c.downcase_mesh_term == term for term in top_terms)
    query = session.query(*cols).filter(filt)
    newdf = pd.read_sql(query.statement, query.session.bind)
    df = list_group(newdf, group_column='nct_id', value_column='interventions')

    # Human-readable names
    human_names = {'interventions': 'list of intervention MESH terms'}

    return df, human_names


def _gather_keywords(conn=None, meta=None, N=10):
    """ Connect to AACT database and gather list of top N most common keywords, 
    grouped by study ID (nct_id)

    Args:
        conn (engine): Sqlalchemy engine connection to database. If None, a new 
            connection will be established. Default is None
        meta (metadata): Sqlalchemy metadata scheme from conn. If None, it will
            be gathered from a new connection    
        N (int): Only keep the top N most common terms/keywords. Default 10

    Return:
        df (DataFrame): pandas dataframe with columns for study ID ('nct_id'), 
            and list of top N intervention mesh terms ('interventions')
        human_names (dict): dictionary mapping columns to human-readable names    
    """
    # Connect to database, if necessary, & open session
    if conn is None or meta is None:
        conn, meta = _connectdb()
    Session = sessionmaker(bind=conn)
    session = Session()

    # Initialize output dataframe
    df = None

    # prep for query
    table = meta.tables['keywords']

    # Gather conditions vocabulary (top N most frequent terms)
    col = table.c.downcase_name.label('keyword')
    countcol = func.count(col).label('count')
    query = session.query(col).group_by(col).order_by(countcol.desc()).limit(N)
    top_terms = [x[0] for x in query.all()]

    # Collect list of top terms in each study
    cols = (table.c.nct_id, table.c.downcase_name.label('keywords'))
    filt = or_(table.c.downcase_name == term for term in top_terms)
    query = session.query(*cols).filter(filt)
    newdf = pd.read_sql(query.statement, query.session.bind)
    df = list_group(newdf, group_column='nct_id', value_column='keywords')

    # Human-readable names
    human_names = {'keywords': 'list of keywords'}

    return df, human_names


def _gather_phase(conn=None, meta=None):
    """ Connect to AACT database and gather phase info

    Kwargs:
        conn (engine): Sqlalchemy engine connection to database. If None, a new 
            connection will be established. Default is None
        meta (metadata): Sqlalchemy metadata scheme from conn. If None, it will
            be gathered from a new connection

    Returns:
        df (dataframe): Pandas dataframe with results, including study id 
            ('nct_id'), and one boolean True/False column for each phase
        human_names (dict): dictionary mapping columns to human-readable names
    """
    # Connect to database, if necessary, & open session
    if conn is None or meta is None:
        conn, meta = _connectdb()
    Session = sessionmaker(bind=conn)
    session = Session()

    # Initialize output dataframe
    df = None

    # query phase info, True if matches that phase, false otherwise
    table = meta.tables['studies']
    for num in range(4):
        filt = table.c.phase.contains(str(num+1))
        cols = (table.c.nct_id, filt.label('phase{}'.format(num+1)))
        query = session.query(*cols).filter(filt)
        newdf = pd.read_sql(query.statement, query.session.bind)
        if df is None:
            df = newdf
        else:
            df = df.merge(newdf, on='nct_id', how='outer')
        df = df.fillna(False)

    # human readable names
    human_names = {'phase1': 'Phase 1',
                   'phase2': 'Phase 2',
                   'phase3': 'Phase 3',
                   'phase4': 'Phase 4'}

    return df, human_names


def _gather_minage(conn=None, meta=None):
    """ Connect to AACT database and gather minimum age for study participants

    Args:
        conn (engine): Sqlalchemy engine connection to database. If None, a new 
            connection will be established. Default is None
        meta (metadata): Sqlalchemy metadata scheme from conn. If None, it will
            be gathered from a new connection    

    Return:
        df (DataFrame): pandas dataframe with columns for study ID ('nct_id'), 
            and minimum age in year ('minage')
        human_names (dict): dictionary mapping columns to human-readable names
    """

    # Connect to database, if necessary, & open session
    if conn is None or meta is None:
        conn, meta = _connectdb()
    Session = sessionmaker(bind=conn)
    session = Session()

    # Initialize output dataframe
    df = None

    # prep for query
    table = meta.tables['calculated_values']

    # Collect age number and units for each study
    cols = (table.c.nct_id, 
            table.c.minimum_age_unit.label('unit'),
            table.c.minimum_age_num.label('num'))
    filt = table.c.minimum_age_num.isnot(None)
    query = session.query(*cols).filter(filt)
    df = pd.read_sql(query.statement, query.session.bind)

    # fix unit case/spelling
    df['unit'] = [re.sub(r's$', '', x).strip() if x is not None
                  else None for x in df['unit'].str.lower()]

    # convert age units into years
    unit_map = {'year': 1., 'month': 1. / 12., 'week': 1. / 52.1429,
                'day': 1. / 365.2422, 'hour': 1. / 8760., 
                'minute': 1. / 525600.}
    df['factor'] = df['unit'].map(unit_map)
    df['minage'] = (df['num'] * df['factor'])

    # Only keep cols of interest
    df = df[['nct_id', 'minage']]

    # humannames
    human_names = {'minage': 'minimum age (years)'}

    return df, human_names


def _gather_otherstuff(conn=None, meta=None):
    """ Connect to AACT database and gather various columns of interest (esp. 
    those that require little processing)

    Args:
        conn (engine): Sqlalchemy engine connection to database. If None, a new 
            connection will be established. Default is None
        meta (metadata): Sqlalchemy metadata scheme from conn. If None, it will
            be gathered from a new connection    

    Return:
        df (DataFrame): pandas dataframe with columns for study ID ('nct_id'), 
            and other features of interest
        human_names (dict): dictionary mapping columns to human-readable names

    Notes:
        - Only return studies that are (1) Compelted and (2) Interventional
    """    

    # Connect to database, if necessary, & open session
    if conn is None or meta is None:
        conn, meta = _connectdb()
    Session = sessionmaker(bind=conn)
    session = Session()

    # Initialize output dataframe
    df = None

    # from calculated values
    table = meta.tables['calculated_values']
    cols = (table.c.nct_id,
            table.c.number_of_facilities.label('facilities'),
            table.c.registered_in_calendar_year.label('year'),
            table.c.actual_duration.label('duration'),
            table.c.has_us_facility.label('usfacility'))
    filt = table.c.actual_duration.isnot(None)
    query = session.query(*cols).filter(filt)
    cdf = pd.read_sql(query.statement, query.session.bind)

    # from interventions
    table = meta.tables['interventions']
    query = session.query(distinct(table.c.intervention_type))
    classes = [x[0] for x in query.all()]
    cols = [table.c.nct_id]
    i_human_names = {}
    for cl in classes:
        clname = cl.lower().replace(' ', '')
        i_human_names[clname] = 'class: {}'.format(cl.lower())
        cols.append(table.c.intervention_type.contains(cl).label(clname))
    query = session.query(*cols)
    idf = pd.read_sql(query.statement, query.session.bind)
    idf = idf.drop_duplicates(subset='nct_id', keep='first')

    # from studies
    table = meta.tables['studies']
    cols = (table.c.nct_id,
            table.c.number_of_arms.label('arms'))
    filt = and_(table.c.study_type.contains('Interventional'),
                table.c.overall_status.contains('Completed'))
    query = session.query(*cols).filter(filt)
    sdf = pd.read_sql(query.statement, query.session.bind)

    # Merge
    df1 = cdf.merge(idf, on='nct_id', how='inner')
    df = sdf.merge(df1, on='nct_id', how='inner')

    # Fill nas
    df['arms'] = df['arms'].fillna(2).astype(int)
    df['facilities'] = df['facilities'].fillna(1).astype(int)
    df['usfacility'] = df['usfacility'].fillna(True).astype(bool)

    # human names
    human_names = {'facilities': 'number of facilities',
                   'year': 'year trial started',
                   'duration': 'study duration (months)',
                   'usfacility': 'has facilities in US',
                   'arms': 'number of arms',
                   **i_human_names}

    return df, human_names


def gather_all_data():
    """ Connect to AACT database and extract data of interest
    """

    # Connect to database
    conn, meta = _connectdb()

    # Gather data
    funcs = (_gather_conditions,
             _gather_interventions,
             _gather_keywords,
             _gather_minage,
             _gather_phase,
             _gather_sex,
             _gather_otherstuff,
             _gather_response)

    df = None
    human_names = {}
    for f in funcs:
        newdf, newhn = f(conn=conn, meta=meta)
        human_names = {**human_names, **newhn}
        if df is None:
            df = newdf
        else:
            df = df.merge(newdf, on='nct_id', how='inner')
    df = df.set_index('nct_id')




def _remove_highdrops(df, thresh=1.0):
    """ Given dataframe, remove rows where the dropout rate is above thresh, and
    return the resulting dataframe
    """
    return df[df['dropped'] < df['enrolled']*thresh]


def get_data(savename=None, dropna=True, N=10, fill_intelligent=True, savename_human=None):
    """ Connect to AACT database and gather data of interest

    Kwargs:
        savename (string): If not None, save the resulting DataFrame with
            data to this file name using pickle
        dropna (bool): If True, drop rows that have any null/nan 
        N (int): For each word-based categorical features (i.e. MeSH conditions,
            MeSH interventions, and keywords), only keep the top N most common 
            strings as features (dummies). Default is 10
        fill_intelligent (bool): If True, fill empty/null/NaNs features with
            best-guess values. If False, leave as NaNs. Default is True
        savename_human (string): If not None, save the resulting human_names 
            dict via pickle

    Return:
        df (DataFrame): Pandas DataFrame with data features and responses
        human_names (dict): dictionary mapping columns to human-readable names        
    """

    # Collect data (features & response, inner join)
    (dfY, Ynames) = _gather_response()
    (dfX, Xnames) = _gather_features(N=N, fill_intelligent=fill_intelligent)
    df = dfY.join(dfX, how='inner').dropna(how='any')

    # human readable names
    human_names = {**Xnames, **Ynames}

    # Remove 100% dropouts
    df = _remove_highdrops(df)

    # Drop columns that only have 1 unique value (no info)
    for c in df.columns.tolist():
        if len(df[c].unique()) < 2:
            df.drop(columns=c, inplace=True)
            del human_names[c]

    # Save dataframe & human_names
    if savename is not None:
        df.to_pickle(savename)

    if savename_human is not None:
        with open(savename_human, 'wb') as output_file:
            pk.dump(human_names, output_file)

    # Return
    return (df, human_names)


def split_data(df, save_suffix=None, test_size=None):
    """ Given data frame, split into training and text sets and save via pickle

    Args:
        df (DataFrame): pandas dataframe with data to split

    Kwargs:
        save_suffix (str): If not none save the training and testing data via 
            pickle, and append this to the filename (e.g. 
            'training_<save_suffix>.pkl' or 'testing_<save_suffix>.pkl'
        test_size (float, int, or None): proportion of data to include in test 
            set, see train_test_split() documentation for more
    
    Returns:
        dfsplit (list): 2-element list with training and testing dataframes, 
            respectively (as output by train_test_split)
    """
    dfsplit = train_test_split(df)

    # Save training and testing data
    if save_suffix is not None:
        dfsplit[0].to_pickle('training_{}.pkl'.format(save_suffix))
        dfsplit[1].to_pickle('testing_{}.pkl'.format(save_suffix))
    
    return dfsplit


def feature_plots(df):
    """ Create & show grid plot and correlation plot for non dummies
    """

    # Select which columns to include (exclude categoricals/dummies)
    cols = []
    for c in df.columns.tolist():
        if len(df[c].unique()) > 3:
            cols.append(c)

    sns.set(font_scale=0.75)

    # GRID PLOT
    tmpdf = df[cols].fillna(False)
    g = sns.PairGrid(tmpdf, size=1)
    g = g.map_diag(plt.hist, bins=50)
    g = g.map_offdiag(plt.scatter, s=5, alpha=0.5)
    # plt.tight_layout()
    plt.show()

    # Corr coefs plot (coerce objects to float)
    cm = df[cols].corr().as_matrix()
    hm = sns.heatmap(cm,
                     cbar=True,
                     robust=True,
                     vmin=-1, vmax=1,
                     center=0,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 7},
                     yticklabels=cols,
                     xticklabels=cols)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def getmodeldata(getnew=False, **kwargs):
    """ Gather data from the 'data' module

    Args:
    getnew (bool): Default False. If True, extract data from scrach. If False, 
        load from file. If True, pass additional kwargs to get_data()

    Returns:
        X (dataframe): Features as a numpy array
        y (dataframe): Response
        human_names (dict): dictionary mapping columns to human-readable names
    """

    # Either gather new data or load from file
    if getnew:
        default_args = {'savename': 'data/full_data.pkl',
                        'savename_human': 'data/human_names.pkl',
                        'N': 50,
                        'dropna': True,
                        'fill_intelligent': True}
        inputargs = {**default_args, **kwargs}
        (df, human_names) = get_data(**inputargs)
        [df, df_test] = split_data(df, save_suffix='data')

    else:
        df = pd.read_pickle('data/training_data.pkl')
        df_test = pd.read_pickle('data/testing_data.pkl')
        with open('data/human_names.pkl', 'rb') as input_file:
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

