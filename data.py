"""
data - a module to extract and process AACT clinical trials data
=================================================================

**data** is a python module for connecting to the AACT relational database
(via PostgreSQL), extract features of interest, and clean/process that data
"""

from config import config
from sqlalchemy import create_engine
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import re
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle as pk


def _connectdb():
    """ Open and return SQLAlchemy engine to PostgreSQL database """

    # read connection parameters
    params = config()

    # connect to the PostgreSQL server
    engine = create_engine('postgresql://%s:%s@%s/%s' %
                           (params['user'], params['password'],
                            params['host'], params['database']))

    return engine


def _gather_response():
    """ Connect to AACT postgres database and collect response variables (number
    of participants enrolled and dropped), with some consistency checks

    Returns:
        df (DataFrame): Pandas dataframe with columns for study ID ('nct_id'), 
            number of participants enrolled ('enrolled') at the start, and the
            number that dropped out ('dropped')
        human_names (dict): dictionary mapping columns to human-readable names

    Notes:
    - Only keep studies with valid (non-nan) data
    - Only keep studies where all of the following are true:
        a. the total number of participants dropped equals the number 'NOT
           COMPLETED' 
        b. the number of participants 'STARTED' equals the number 'COMPLETED'
           plus the number 'NOT COMPLETED'
        c. the number of participants 'STARTED' equals the number 'enrolled'
    """

    # Connect to AACT database
    engine = _connectdb()


    # Gather enrollment/dropout numbers - PART 1a
    #   Gather dropout info from the 'drop_withdrawals' table by summing
    #   the total count of people that dropped out within each study
    colnames = {'nct_id': 'nct_id', 
                'count':'dropped'}
    df = pd.read_sql_table('drop_withdrawals', engine,
                           columns=colnames.keys()
                           ).groupby('nct_id').sum().rename(columns=colnames)

    # Gather enrollment/dropout numbers - PART 1b
    #   Gather enrollment numbers (actual, not anticipated) from 'studies' table
    #   and append to existing dataframe
    colnames = {'nct_id':'nct_id', 
                'enrollment':'enrolled', 
                'enrollment_type': 'enrollment_type'}
    studies = pd.read_sql_table('studies', engine, 
                                columns=colnames.keys()
                                ).set_index('nct_id').rename(columns=colnames)
    filt = [x=='Actual' for x in studies['enrollment_type']]
    df = df.join(studies[filt][['enrolled']].astype(int), how='inner')
    df.dropna(how='any', inplace=True)

    # Gather enrollment/dropout numbers - PART 2
    #   Gather enrollment and dropout numbers from the 'milestones' table, only
    #   looking at the COMPLTED/STARTED/NOT COMPLETED counts, and append to 
    #   existing dataframe
    colnames = {'nct_id': 'nct_id', 
                'title': 'milestone_title', 
                'count':'milestone_count'}
    df2 = pd.read_sql_table('milestones', engine, columns=colnames.keys())
    value_str = ['COMPLETED', 'STARTED', 'NOT COMPLETED']
    for s in value_str:
        filt = df2['title'].str.match(s)
        df = df.join(df2[filt][['nct_id','count']] \
            .groupby('nct_id').sum().rename(columns={'count':s}), how='inner')

    # Check the various enrollment measures against each other and only keep 
    # studies that make sense
    filt = ((df['enrolled'] == df['STARTED']) & 
            (df['dropped'] == df['NOT COMPLETED']) &
            (df['STARTED'] == (df['NOT COMPLETED']+df['COMPLETED'])))
    df = df[filt]

    # return limited dataframe
    df = df[['enrolled', 'dropped']]

    # human-readable names
    human_names = {'enrolled': 'number of participants enrolled',
                   'enrolled': 'number participants dropped'}

    return (df, human_names)


def _gather_features(N=10, fill_intelligent=True):
    """ Connect to AACT database, join select data, and return as a dataframe

    Args:
        N (int): For each word-based categorical features (i.e. MeSH conditions,
            MeSH interventions, and keywords), only keep the top N most common 
            strings as features (dummies). Default is 10
        fill_intelligent (bool): If True, fill empty/null/NaNs with best-guess 
            values. If False, leave as NaNs. Default is True

    Return:
        df (DataFrame): pandas dataframe with full data
        human_names (dict): dictionary mapping columns to human-readable names

    Notes:
    - filter for Completed & Inverventional studies only
    - Creates dummy variables
    """

    """ Notes to self about tables
    table_names = [
        # 'baseline_counts',              # x
        'baseline_measurements',        # Y male/female [category, param_value_num]
        # 'brief_summaries',              # ~ long text description
        'browse_conditions',            # Y mesh terms of disease (3700) -> heirarchy, ID --> Get this!
        'browse_interventions',         # Y mesh terms of treatment (~3000)
        'calculated_values',            # Y [number_of_facilities, registered_in_calendar_year, registered_in_calendar_year, registered_in_calendar_year, min age, max age]
        # 'conditions',                   # x condition name
        # 'countries',                    # ~ Country name
        # 'design_group_interventions',   # x
        # 'design_groups'                 # x
        # 'design_outcomes',              # x
        # 'designs',                      # x~ subject/caregiver/investigator blinded?
        # 'detailed_descriptions',        # x 
        # 'drop_withdrawals',             # Y --> already in response
        # 'eligibilities',                # Y (genders) --> Already got from baseline?
        # 'facilities',                   # x
        # 'intervention_other_names',     # x
        'interventions',                # Y intervetion_type (11)
        'keywords',                     # Y downcase_name (160,000!)
        # 'milestones',                   # Y title (NOT COMPLETE/COMPLETED, 90,000) and count --> already in response
        # 'outcomes',                     # x
        # 'participant_flows',            # x
        # 'reported_events',              # x
        # 'result_groups',                # x
        'studies'                       # Y [study_type, overall_status (filt), phase (parse), number_of_arms, number_of_groups, has_dmc, is_fda_regulated_drug, is_fda_regulated_device, is_unapproved_device]
    ]
    """

    # Prep args
    N = int(N)

    # Connect to database
    engine = _connectdb()

    # ================ Gather fe/male counts from 'baseline_measurements'
    colnames = {'nct_id': 'nct_id',
                'category': 'category',
                'classification': 'classification',
                'param_value_num': 'count'}
    meas = pd.read_sql_table('baseline_measurements', engine,
                             columns=colnames.keys()).rename(columns=colnames)
    meas.set_index('nct_id', inplace=True)

    # Determine if these particpant group counts are for fe/male
    sexes = ['male', 'female']
    for s in sexes:
        filt = ((meas['category'].str.lower().str.match(s) |
                 meas['classification'].str.lower().str.match(s)) &
                meas['count'].notnull())
        if fill_intelligent:
            meas[s] = int(0)
        else:
            meas[s] = np.nan
        meas.loc[filt, s] = meas[filt]['count']

    # Group/sum by study id, forcing those with no info back to nans
    noinfo = meas[sexes].groupby('nct_id').apply(lambda x: True if np.all(np.isnan(x)) else False)
    meas = meas[sexes].groupby('nct_id').sum()

    # Convert to fraction male
    meas['malefraction'] = meas['male']/(meas['female']+meas['male'])
    meas.drop(columns=sexes, inplace=True)

    if fill_intelligent:
        meas['malefraction'] = meas['malefraction'].fillna(0.5).astype(float) # asumme 50/50 split
    else:
        meas.loc[noinfo, 'malefraction'] = np.NaN

    # human-readable names
    meas_humannames = {'malefraction': 'fraction of males'}

    # ================ Gather condition MeSH terms from 'browse_conditions'
    colnames = {'nct_id': 'nct_id',
                'mesh_term': 'cond'}
    conds = pd.read_sql_table('browse_conditions', engine,
                              columns=colnames.keys()
                              ).rename(columns=colnames).set_index('nct_id')
    conds['cond'] = conds['cond'].str.lower()

    # Limit to the to N terms & create dummy vars
    topN_conds = conds['cond'].value_counts().head(N).index.tolist()    

    conds['cond'] = [re.sub(r'[^a-z]', '', x) if x in topN_conds
                     else None for x in conds['cond']]
    conds = pd.get_dummies(conds).groupby('nct_id').any()

    # human readable names
    prefix = colnames['mesh_term']
    topN_conds_colnames = []
    for c in topN_conds:
        dummyname = prefix + '_' + re.sub(r'[^a-z]', '', c)
        topN_conds_colnames.append(dummyname)

    conds_humannames = dict(zip(topN_conds_colnames, topN_conds))

    # ================ Gather intervention MeSH terms in 'browse_interventions'
    colnames = {'nct_id': 'nct_id',
                'mesh_term': 'intv'}    
    intv = pd.read_sql_table('browse_interventions', engine,
                             columns=colnames.keys()
                             ).rename(columns=colnames).set_index('nct_id')
    intv['intv'] = intv['intv'].str.lower()

    # Limit to the to N terms & create dummy vars
    topN_intv = intv['intv'].value_counts().head(N).index.tolist()
    intv['intv'] = [re.sub(r'[^a-z]', '', x) if x in topN_intv 
                    else None for x in intv['intv']]
    intv = pd.get_dummies(intv).groupby('nct_id').any()

    # human readable names
    prefix = colnames['mesh_term']
    topN_intv_colnames = []
    for c in topN_intv:
        dummyname = prefix + '_' + re.sub(r'[^a-z]', '', c)
        topN_intv_colnames.append(dummyname)

    intv_humannames = dict(zip(topN_intv_colnames, topN_intv))

    # ================ Gather various info from 'calculated_values'  
    colnames = {'nct_id': 'nct_id',
                'number_of_facilities': 'facilities',
                'registered_in_calendar_year': 'year',
                'actual_duration': 'duration',
                'has_us_facility': 'usfacility',
                'minimum_age_num': 'minimum_age_num',
                'minimum_age_unit': 'minimum_age_unit'}
    calc = pd.read_sql_table('calculated_values', engine,
                             columns=colnames.keys()
                             ).rename(columns=colnames).set_index('nct_id')

    # convert age units into years
    unit_map = {'year': 1., 'month':1/12., 'week': 1/52.1429,
                'day': 1/365.2422, 'hour': 1/8760., 'minute': 1/525600.}

    calc['minimum_age_unit'] = [re.sub(r's$', '', x).strip() if x is not None
                                else None for x in
                                calc['minimum_age_unit'].str.lower()]
    calc['minimum_age_factor'] = calc['minimum_age_unit'].map(unit_map)
    calc['minimum_age_years'] = (calc['minimum_age_num'] *
                                 calc['minimum_age_factor'])

    # only keep colums we need, & rename some
    colnames2 = {'facilities': 'facilities',
                'year': 'year',
                'duration': 'duration',
                'usfacility': 'usfacility',
                'minimum_age_years': 'minage'} # removing maxage - mostly empty not useful
    calc = calc[list(colnames2.keys())].rename(columns=colnames2)
    
    # Fill nans with best-guesses
    if fill_intelligent:
        calc['minage'] = calc['minage'].fillna(0).astype(int) # assume minage is 0
        calc['usfacility'] = calc['usfacility'].fillna(True) # assume yes it has US facility
        calc['facilities'] = calc['facilities'].fillna(1).astype(int) # assume 1 facility

    # human-readable names
    calc_humannames = {colnames2[colnames['number_of_facilities']]: 'number of facilities',
                       colnames2[colnames['registered_in_calendar_year']]: 'calendar year trial started',
                       colnames2[colnames['actual_duration']]: 'study duration (months)',
                       colnames2[colnames['has_us_facility']]: 'at least one facility in the US',
                       colnames2['minimum_age_years']: 'minimum age (years)' }

    # ================ Gather intervention type info from 'interventions' 
    colnames = {'nct_id': 'nct_id',
                'intervention_type': 'intvtype'}
    intvtype = pd.read_sql_table('interventions', engine,
                             columns=colnames.keys()
                             ).rename(columns=colnames).set_index('nct_id')
    
    # drop duplicates
    intvtype = intvtype[~intvtype.index.duplicated(keep='first')]

    # convert to lowercase, remove non-alphabetic characters
    topN_intvtype = intvtype['intvtype'].unique().tolist()
    intvtype['intvtype'] = [re.sub(r'[^a-z]', '', x)
                            for x in intvtype['intvtype'].str.lower()]
    intvtype = pd.get_dummies(intvtype).groupby('nct_id').any()

    # human-readable names
    prefix = colnames['intervention_type']
    topN_intvtype_colnames = []
    for c in topN_intvtype:
        dummyname = prefix + '_' + re.sub(r'[^a-z]', '', c.lower())
        topN_intvtype_colnames.append(dummyname)

    intvtype_humannames = dict(zip(topN_intvtype_colnames, topN_intvtype))


    # ================ Gather keywords info from 'keywords' (only keep top N)
    colnames = {'nct_id': 'nct_id',
                'name': 'keyword'}
    words = pd.read_sql_table('keywords', engine,
                              columns=colnames.keys()
                              ).rename(columns=colnames).set_index('nct_id')
    words['keyword'] = words['keyword'].str.lower()

    # Limit to the to N terms & create dummy vars
    topN_words = words['keyword'].value_counts().head(N).index.tolist()
    words['keyword'] = [re.sub(r'[^a-z]', '', x) if x in topN_words
                        else None for x in words['keyword']]
    words = pd.get_dummies(words).groupby('nct_id').any()

    # human-readable names
    prefix = colnames['name']
    topN_words_colnames = []
    for c in topN_words:
        dummyname = prefix + '_' + re.sub(r'[^a-z]', '', c)
        topN_words_colnames.append(dummyname)

    words_humannames = dict(zip(topN_words_colnames, topN_words))

    # ================ Gather various info from 'studies' (filter for Completed & Inverventional studies only!)
    colnames = {'nct_id': 'nct_id',
                'study_type': 'studytype',
                'overall_status': 'status',
                'phase': 'phase',
                'number_of_arms': 'arms'}
    studies = pd.read_sql_table('studies', engine,
                                columns=colnames.keys()
                                ).rename(columns=colnames).set_index('nct_id')
    
    # filter to only keep 'Completed' studies
    filt = (studies['status'].str.match('Completed') & 
            studies['studytype'].str.match('Interventional'))
    studies = studies[filt].drop(columns=['status', 'studytype'])

    # parse study phases
    for n in [1,2,3, 4]:
        filt = studies['phase'].str.contains(str(n))
        studies['phase'+str(n)] = False
        studies.loc[filt,'phase'+str(n)] = True
    studies.drop(columns=['phase'], inplace=True)

    if fill_intelligent:
        studies['arms'] = studies['arms'].fillna(1).astype(int)

    # human-readable names
    studies_humannames = {colnames['number_of_arms']: 'number of study arms',
                          'phase1': 'phase 1',
                          'phase2': 'phase 2',
                          'phase3': 'phase 3',
                          'phase4': 'phase 4'}

    # ================ Combine all dataframes together!
    # Note: inner join all data onto 'studies' (so only keep data for completed, 
    # interventional studies)

    df = studies
    for d in [meas, conds, intv, calc, intvtype, words]:
        df = df.join(d, how='inner')

    # join all human-readable names
    human_names = {**studies_humannames,
            **meas_humannames,
            **conds_humannames,
            **intv_humannames,
            **calc_humannames,
            **intvtype_humannames,
            **words_humannames}

    return (df, human_names)


def remove_highdrops(df, thresh=1.0):
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
    (dfX, Xnames) = _gather_features(N=N, fill_intelligent=fill_intelligent)
    (dfY, Ynames) = _gather_response()
    df = dfY.join(dfX, how='inner').dropna(how='any')

    # human readable names
    human_names = {**Xnames, **Ynames}

    # Remove 100% dropouts
    df = remove_highdrops(df)

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


