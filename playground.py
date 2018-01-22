import demo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============ Correlation plots, etc
df = demo.get_data()

cols = []
for c in list(df.columns):
    if c.find('_')<0 and c.find('phase')<0 and c.find('groups')<0:
        cols.append(c)

sns.pairplot(df[cols], dropna=True
plt.show()

# ============ GATHER DATA
df = pd.read_pickle('training_data.pkl')


# ============ Compare complete/not complete vs Total dropouts



import demo
import pandas as pd
engine = demo.connectdb()

# V1
colnames = {'nct_id': 'nct_id', 'count':'dropped'}
df = pd.read_sql_table('drop_withdrawals', engine,
                       columns=colnames.keys()
                       ).groupby('nct_id').sum().rename(columns=colnames)

# V2
colnames = {'nct_id': 'nct_id', 'title': 'milestone_title', 'count':'milestone_count'}
df2 = pd.read_sql_table('milestones', engine, columns=colnames.keys())

value_str = ['COMPLETED', 'STARTED', 'NOT COMPLETED']
for s in value_str:
    filt = df2['title'].str.match(s)
    df = df.join(df2[filt][['nct_id', 'count']].groupby('nct_id').sum().rename(columns={'count':s}),
                 how='inner')

colnames = {'nct_id':'nct_id', 'enrollment':'enrolled', 'enrollment_type': 'enrollment_type'}
studies = pd.read_sql_table('studies', engine, 
                            columns=colnames.keys()
                            ).set_index('nct_id').rename(columns=colnames)
filt = [x=='Actual' for x in studies['enrollment_type']]
df = df.join(studies[filt][['enrolled']].astype(int), how='inner')
df.dropna(how='any', inplace=True)

# only keep rows that make sense
filt = ((df['enrolled'] == df['STARTED']) & 
        (df['dropped'] == df['NOT COMPLETED']) &
        (df['STARTED'] == (df['NOT COMPLETED']+df['COMPLETED'])))
df = df[filt]







# ============= FIT MODEL
res = demo.fit_model(df)

# Show results
print(res.summary())
demo.diagnotic_plots(res, show=True)

# Save linear model results via pickle (data included)
res.save('training_res_mod.pkl', remove_data=False)




# PLOTS
sns.set(style='whitegrid', context='notebook')
cols = ['start_year', 'duration', 'num_facilities', 'minimum_age_years', 'enrolled','droprate']
df2 = df.dropna()
sns.pairplot(df2[cols], size=2.5)
plt.show()



cols = df.columns
cm = np.corrcoef(df2[cols].values.T)
fig = sns.set(font_scale=0.75)
hm = sns.heatmap(cm,
    cbar=True,
    annot=True,
    square=True,
    fmt='.2f',
    annot_kws={'size': 15},
    yticklabels=cols,
    xticklabels=cols)
plt.show()






# =============  FIGURES FOR WEEK2 DEMO
# Histogram of dropout rate
fig, ax = plt.subplots(figsize=(6,3))
filt = df['is_cancer']==False
sns.distplot(df[filt]['droprate'], bins=50, kde=False, label='not cancer', norm_hist=True)
ax.set(yticks=[], xlabel='dropout rate (fraction)')
fig.tight_layout()
sns.despine(left=True)
fig.show()

# Dropout rate +/- cancer
fig, ax = plt.subplots(figsize=(6,3))
filt = df['is_cancer']==False
sns.distplot(df[filt]['droprate'], bins=50, kde=False, label='not cancer', norm_hist=True)
filt = df['is_cancer']==True
sns.distplot(df[filt]['droprate'], bins=50, kde=False, label='cancer', norm_hist=True)
ax.set(yticks=[], xlabel='dropout rate (fraction)')
fig.tight_layout()
sns.despine(left=True)
plt.legend()
fig.show()

# Dropout rate vs study duration
sns.set_style("white")  
fig, ax = plt.subplots(figsize=(6,3))
sns.regplot(x='duration', y='droprate', data=df, scatter_kws={'s':2, 'alpha':0.5}, fit_reg=False)
ax.set(xlabel='study duration (months)', ylabel='dropout rate (fraction)',
       xlim=(0,150), yticks=[0,1])
fig.tight_layout()
sns.despine()
fig.show()

# log plot?
fig, ax = plt.subplots(figsize=(6,3))
plt.semilogx(df['duration'], df['droprate'], '.', alpha=0.1)
ax.set(xlabel='study duration (months)', ylabel='dropout rate (fraction)',
       xlim=(0,150), yticks=[0,1])
fig.tight_layout()
sns.despine()
fig.show()

# ============= EVALUATE MODEL

# Load test data set
dftest = pd.read_pickle('testing_data.pkl')


# sklearn getdummies


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
