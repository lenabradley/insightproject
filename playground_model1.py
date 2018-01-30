# -*- coding: utf-8 -*-
import data
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pickle as pk
import pandas as pd
import re
import numpy as np
import seaborn as sns


# ===============================================================
# GET/MAKE DATA AND METADATA
# ===============================================================
(Xraw, yraw, human_names) = data.getmodeldata(getnew=False)

feature_names = Xraw.columns.tolist()
response_names = yraw.columns.tolist()

X = Xraw.as_matrix()
y = yraw[response_names[0]].as_matrix()
ytform = y**(1/3)


# === Feature metadata
# Establish dataframe with column metadata, including (1) if it is continuous 
# and (2) it's human-readable name
column_info_dict = {}
for (k, v) in human_names.items():
    if k in Xraw:
        iscat = False
        if len(Xraw[k].unique()) < 3:
            iscat = True
        column_info_dict[k] = {'name':v, 'categorical': iscat}

column_info = pd.DataFrame(column_info_dict).T
column_info.index.rename('colname', inplace=True)


# Collect which terms (i.e. categorical dummies) are for each of these groups:
#   (1) condition mesh terms
#   (2) intervention mesh terms
#   (3) intervention type
#   (4) keywords
#   (5) phase
prefixes = ['cond_', 'intv_', 'intvtype_', 'keyword_', 'phase'] # i.e. groups
for p in prefixes:
    # Regex to check if prefix is at the begining of a string (i.e. column name)
    re_search_str = r'^{}'.format(p)
    column_info['is_'+p] = False

    # look at each column name, if it matches the prefix, indicate that
    for n in column_info.index.tolist():
        if re.search(re_search_str, n) is not None:
            column_info.loc[n, 'is_'+p] = True


# ===============================================================
# LINEAR MODEL (LASSO + TFORM Y + NORMALIZATION)
# ===============================================================

# === Initialize model
reg = linear_model.Lasso(normalize=True)

# reg.fit(X, ytform)
# ypred = reg.predict(X)

# # === PRINT OUTPUTS === #
# print('\n ** Linear regression + normalization + transform y + LASSO')
# print('Non-zero coefficients:')
# for (c,f) in sorted(zip(reg.coef_, feature_names)):
#     if abs(c) > 0:
#         print('{:+0.2f}\t{}'.format(c, f))

# print("RMS error: {:.2f}".format(mean_squared_error(ytform, ypred)**(1/2)))
# print('Training R2 score: {:.2f}'.format(r2_score(ytform, ypred)))


# === CV grid search for hyperparameters 
clf = GridSearchCV(reg, 
             param_grid={'alpha': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]},
             scoring=make_scorer(r2_score))
clf.fit(X, ytform)
print("Best hyperparameters: {}".format(clf.best_params_))
print("* Grid scores:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("  %0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
best_params = clf.best_params_


# === Fit with best alpha
reg = linear_model.Lasso(**best_params, normalize=True)
reg.fit(X, ytform)
ypred = reg.predict(X)


# === Print output
print('\n ** Linear regression + normalization + transform y + LASSO')
print('Non-zero coefficients:')
for (c,f) in sorted(zip(reg.coef_, feature_names)):
    if abs(c) > 1e-4:
        print('{:+0.2f}\t{}'.format(c, f))
print("RMS error: {:.2f}".format(mean_squared_error(ytform, ypred)**(1/2)))
print('Training R2 score: {:.2f}'.format(r2_score(ytform, ypred)))




indices = np.argsort(np.abs(reg.coef_))[::-1]
ranked_coefs = reg.coef_[indices]
ranked_names = [column_info.loc[feature_names[j], 'name'] for j in indices]

print(' ')
print('Coefficients by effect size:')
for (c,f) in zip(ranked_coefs, ranked_names):
    if abs(c) > 1e-4:
        print('{:+0.2f}\t{}'.format(c, f))



for (c, f) in sorted(zip(reg.coef_, feature_names)):
    if abs(c) > 1e-4:
        print('{:+0.2f}\t{}'.format(c, f))



# === MODEL: K-FOLD CROSS VALIDATION (i.e. check generalizable) 
nfolds = 10
CV_scores = cross_val_score(reg, X, ytform,
                            cv=nfolds, 
                            scoring=make_scorer(r2_score))
print('{:d}-fold CV R2 score: {:0.2f}+/-{:0.2f}'
      .format(nfolds, CV_scores.mean(), CV_scores.std()))


# === Plot learning curves

train_sizes, train_scores, test_scores = \
    learning_curve(reg, X, ytform, cv=None, scoring=make_scorer(r2_score))

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
plt.ylabel('R2')
plt.xlabel('Training examples')
plt.ylim((-0.5, 1))
plt.legend(loc="best")
plt.show()


# === PLOT RESIDUALS 

sns.set(font_scale=1.5, style='white')
ypred = reg.predict(X)

fig = plt.figure(figsize=(10, 5))

# hist of resids
plt.subplot(1, 2, 1)
sns.distplot(ytform-ypred, bins=50, kde=False, vertical=True)
sns.despine(fig=fig, bottom=True, left=True)
plt.xticks([])
plt.ylabel('Residuals')
plt.ylim((-0.6, 0.6))

# residuals vs predicted
plt.subplot(1, 2, 2)
sns.regplot(ypred, ytform-ypred, fit_reg=False, scatter_kws={'alpha': 0.2})
plt.ylim((-0.6, 0.6))
sns.despine()
plt.xlabel('predicted')
plt.show()



# === CALCULATE TEST SET SCORE

feature_names = Xraw.columns.tolist()
# response_names = yraw.columns.tolist()
response_names = ['dropped', 'enrolled']

dftest = pd.read_pickle('testing_data.pkl')
Xtest_raw = dftest[feature_names]
ytest_raw = dftest[response_names]
ytest_raw['droprate'] = ytest_raw['dropped']/ytest_raw['enrolled']

Xtest = Xtest_raw.as_matrix()
ytest = ytest_raw[['droprate']].as_matrix()


r2_test = reg.score(Xtest, ytest**(1/3))

print(r2_test)



# === SAVE DATA & MODEL & METADATA VIA PICKEL === #

# DATA
filename = 'app_data/Xraw_data.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(Xraw, output_file)

filename = 'app_data/yraw_data.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(yraw, output_file)

filename = 'app_data/X_data.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(X, output_file)

filename = 'app_data/y_data.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(y, output_file)

# MODEL
filename = 'app_data/reg_data.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(reg, output_file)

# METADATA
filename = 'app_data/human_names.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(human_names, output_file)

filename = 'app_data/column_info.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(column_info, output_file)

