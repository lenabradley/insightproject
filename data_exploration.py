import model1
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt



# ===============================================================
# GET DATA AND METADATA
# ===============================================================
(Xraw, yraw, human_names) = model1.getmodeldata(getnew=False)

feature_names = Xraw.columns.tolist()
response_names = yraw.columns.tolist()

X = Xraw.as_matrix()
y = yraw[response_names[0]].as_matrix()

with open('app_model1/column_info.pkl', 'rb') as input_file:
    column_info = pk.load(input_file)
column_info['name'] = [x.capitalize() for x in column_info['name']]


# ===============================================================
# EDA
# ===============================================================

# === Histogram of response
df = pd.read_pickle('rawdata.pkl')
droprate = df['dropped']/df['enrolled']
sns.set(style='white', font_scale=1)
sns.distplot(droprate, kde=False)
plt.yticks([])
sns.despine(left=True)
plt.xlabel('Dropout rate (fraction)')
plt.show()


# === Feature correlations

filt = (column_info['categorical']==False) | (column_info['is_intvtype_']==True)
# filt = (column_info['categorical']==False) | (column_info['categorical']==True)
keepcols = column_info[filt].index
hnames = column_info[filt]['name'].tolist()

# Make plot
sns.set(font_scale=0.75, style='white')
hm = sns.heatmap(Xraw[keepcols].corr().as_matrix(),
    cbar=True,
    annot=False,
    square=True,
    fmt='.1f',
    annot_kws={'size': 7},
    yticklabels=hnames, xticklabels=hnames,
    vmin=-1, vmax=1,
    center=0)
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
plt.tight_layout()
plt.show()
