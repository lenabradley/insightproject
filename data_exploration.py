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


# === Cluster map of feature correlations

filt = (column_info['categorical']==False) | (column_info['is_intvtype_']==True)

keepcols = column_info[filt].index

hnames = column_info[filt]['name'].tolist()
hm_args = {'xticklabels': hnames, 'yticklabels': hnames, 
    'vim': -1, 'vmax': 1, 'center': 0}




sns.heatmap(Xraw[keepcols].corr(),
    xticklabels=hnames, yticklabels=hnames,
    vmin=-1, vmax=1, center=0)
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
plt.tight_layout()
plt.show()
