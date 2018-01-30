# -*- coding: utf-8 -*-
import data
# from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV
import matplotlib.pyplot as plt
import pickle as pk
import pandas as pd
# import re
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from skgarden import RandomForestQuantileRegressor

# ===============================================================
# GET DATA AND METADATA
# ===============================================================
(Xraw, yraw, human_names) = data.getmodeldata(getnew=False)

feature_names = Xraw.columns.tolist()
response_names = yraw.columns.tolist()

X = Xraw.as_matrix()
y = yraw[response_names[0]].as_matrix()

with open('data/column_info.pkl', 'rb') as input_file:
    column_info = pk.load(input_file)
column_info['name'] = [x.capitalize() for x in column_info['name']]



# ===============================================================
# DECISION TREE
# ===============================================================

# setup regressor
reg = DecisionTreeRegressor()

# grid search for params
clf = GridSearchCV(reg, 
             param_grid={'max_depth': [None, 2, 3, 4, 5, 6, 7, 8],
                         'max_features': [None, 5, 10, 50, 100]},
             scoring=make_scorer(r2_score))
clf.fit(X, y)
print("Best hyperparameters: {}".format(clf.best_params_))
print("* Grid scores:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("  %0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
best_params = clf.best_params_
print(best_params)


# fit decision tree
reg = DecisionTreeRegressor(**best_params)
reg.fit(X, y)
r2_train = reg.score(X, y)
print('Training data R2 score: {:0.2f}'.format(r2_train))

# === KFOLD CROSS VALIDATION (R2)
nfolds = 10
CV_scores = cross_val_score(reg, X, y,
                            cv=nfolds, 
                            scoring=make_scorer(r2_score))
print('{:d}-fold CV R2 score: {:0.2f}+/-{:0.2f}'
      .format(nfolds, CV_scores.mean(), CV_scores.std()))


# VIZUALIZE decision tree
from sklearn import tree
tree.export_graphviz(reg, out_file='tree.dot')
dot_data = tree.export_graphviz(reg, out_file='tree.dot', 
                         feature_names=Xraw.columns,  
                         filled=True, rounded=True) 
# dot -Tpng tree.dot -o tree.png    (PNG format)
# dot -Tps tree.dot -o tree.ps    (PS format)


# === PLOT RESIDUALS
sns.set(font_scale=1.5, style='white')
ypred = reg.predict(X)

fig = plt.figure(figsize=(10, 5))

# hist of resids
plt.subplot(1, 2, 1)
sns.distplot(y-ypred, bins=50, kde=False, vertical=True)
sns.despine(fig=fig, bottom=True, left=True)
plt.xticks([])
plt.ylabel('Residuals')
plt.ylim((-0.6, 0.6))

# residuals vs predicted
plt.subplot(1, 2, 2)
sns.regplot(ypred, y-ypred, fit_reg=False, scatter_kws={'alpha': 0.2})
plt.ylim((-0.6, 0.6))
sns.despine()
plt.xlabel('predicted')
plt.show()



# === LEARNING CURVES 
train_sizes, train_scores, test_scores = \
    learning_curve(reg, X, y,
                   cv=None, scoring=make_scorer(r2_score))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

sns.despine()
plt.ylabel('R2 score')
plt.xlabel('Training examples')
plt.ylim((0, 1))
plt.legend(loc="best")
plt.show()


# ===============================================================
# COMPUTE TEST SET SCORE - DT
# ===============================================================

feature_names = Xraw.columns.tolist()

dftest = pd.read_pickle('data/testing_data.pkl')
Xtest_raw = dftest[feature_names]
ytest_raw = dftest[['dropped', 'enrolled']]
ytest_raw['droprate'] = ytest_raw['dropped']/ytest_raw['enrolled']

Xtest = Xtest_raw.as_matrix()
ytest = ytest_raw[['droprate']].as_matrix()


r2_test = reg.score(Xtest, ytest)


print('R2 test score:', r2_test)


# ===============================================================
# RANDOM FOREST REGRESSOR
# ===============================================================

# === SETUP RANDOM FOREST
reg = RandomForestRegressor()
nfolds = 10


# === GRID SEARCH FOR HYPERPARAMETERS
clf = GridSearchCV(reg, 
             param_grid={'n_estimators': [100, 1000],
                         'max_depth': [10],
                         'max_features': [None],
                         'min_samples_leaf': [5]},
             scoring=make_scorer(r2_score), cv=nfolds)
clf.fit(X, y)

print("* Grid scores:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("  %0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

best_params = clf.best_params_
print("Best hyperparameters: {}".format(best_params))


# === FIT RF-REGRESSION WITH BEST PARAMS
best_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 5}

reg = RandomForestRegressor(**best_params)
reg.fit(X, y)

# print report
r2_train = reg.score(X, y)
print('Training data R2 score: {:0.2f}'.format(r2_train))


# === KFOLD CROSS VALIDATION (R2)
CV_scores = cross_val_score(reg, X, y,
                            cv=nfolds, 
                            scoring=make_scorer(r2_score))
print('{:d}-fold CV R2 score: {:0.2f}+/-{:0.2f}'
      .format(nfolds, CV_scores.mean(), CV_scores.std()))


# === PLOT RESIDUALS
sns.set(font_scale=1.5, style='white')
ypred = reg.predict(X)

fig = plt.figure(figsize=(10, 5))

# hist of resids
plt.subplot(1, 2, 1)
sns.distplot(y-ypred, bins=50, kde=False, vertical=True)
sns.despine(fig=fig, bottom=True, left=True)
plt.xticks([])
plt.ylabel('Residuals')
plt.ylim((-0.6, 0.6))

# residuals vs predicted
plt.subplot(1, 2, 2)
sns.regplot(ypred, y-ypred, fit_reg=False, scatter_kws={'alpha': 0.2})
plt.ylim((-0.6, 0.6))
sns.despine()
plt.xlabel('predicted')
plt.show()


# === FEATURE IMPORTANCES & PLOT

# Calculate feature importances
names = Xraw.columns.tolist()
importances = reg.feature_importances_
std = np.std([tree.feature_importances_ for tree in reg.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


ranked_importances = importances[indices]
ranked_std = std[indices]
ranked_names = [names[j] for j in indices]

column_info.loc['completed', 'name'] = 'Number of participants needed'

ranked_humannames = []
for n in ranked_names:
    hname = column_info.loc[n, 'name']
    if column_info.loc[n, 'is_cond_']:
        hname = 'Condition: ' + hname
    elif column_info.loc[n, 'is_intv_']:
        hname = 'Intervention: ' + hname
    elif column_info.loc[n, 'is_intvtype_']:
        hname = 'Class: ' + hname
    elif column_info.loc[n, 'is_keyword_']:
        hname = 'Keyword: ' + hname
    ranked_humannames.append(hname)


# Plot the feature importances of the regression (top N)
N = 15
sns.set(style='whitegrid')
f, ax = plt.subplots(figsize=(6,4))
sns.barplot(ranked_importances[:N], 
            ranked_humannames[:N],
            ci=ranked_std[:N])
plt.title("Feature importance")
plt.tight_layout()
plt.show()


# === LEARNING CURVES 
train_sizes, train_scores, test_scores = \
    learning_curve(reg, X, y,
                   cv=nfolds, scoring=make_scorer(r2_score))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

sns.despine()
plt.ylabel('R2 score')
plt.xlabel('Training examples')
plt.ylim((0, 1))
plt.legend(loc="best")
plt.show()


# ===============================================================
# RF QUANTILE REGRESSOR
# ===============================================================

# == fit
rfqr = RandomForestQuantileRegressor(**best_params)
rfqr.fit(X, y)
lower = rfqr.predict(X, quantile=2.5)
upper = rfqr.predict(X, quantile=97.5)
med = rfqr.predict(X, quantile=50)
ypred = reg.predict(X)

# plot confidence intervals
sort_ind = np.argsort(ypred)
plt.plot(np.arange(len(upper)), lower[sort_ind], label='lower')
plt.plot(np.arange(len(upper)), ypred[sort_ind], label='predicted')
plt.plot(np.arange(len(upper)), med[sort_ind], label='median')
plt.plot(np.arange(len(upper)), upper[sort_ind], label='upper')
plt.xlabel('ordered samples')
plt.ylabel('dropout rate')
plt.legend()
plt.show()


# ===============================================================
# SAVE MODEL
# ===============================================================

# MODEL
filename = 'data/reg_model2.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(reg, output_file)


# QUANTILE MODEL 
filename = 'data/reg_model2_quantile.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(rfqr, output_file)


# ===============================================================
# COMPUTE TEST SET SCORE - RF
# ===============================================================

feature_names = Xraw.columns.tolist()

dftest = pd.read_pickle('data/testing_data.pkl')
Xtest_raw = dftest[feature_names]
ytest_raw = dftest[['dropped', 'enrolled']]
ytest_raw['droprate'] = ytest_raw['dropped'] / ytest_raw['enrolled']

Xtest = Xtest_raw.as_matrix()
ytest = ytest_raw[['droprate']].as_matrix()


r2_test = reg.score(Xtest, ytest)


print('R2 test score:', r2_test)




