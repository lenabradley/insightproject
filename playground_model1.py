# -*- coding: utf-8 -*-
import model1
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pickle as pk
import pandas as pd
import re
import numpy as np


# === GET DATA === #
(Xraw, yraw, human_names) = model1.getmodeldata(getnew=False)

feature_names = Xraw.columns.tolist()
response_names = yraw.columns.tolist()

X = Xraw.as_matrix()
y = yraw[response_names[0]].as_matrix()
ytform = y**(1/3)


# === GET FEATURE METADAA ===#

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

# Collect list of terms (i.e. categorial dummies) for ...
#   (1) condition mesh terms
#   (2) intervention mesh terms
#   (3) intervention type
#   (4) keywords
#   (5) phase
# ... also add T/F is_term to column metadata
column_info['isterm'] = False
prefixes = ['cond_', 'intv_', 'intvtype_', 'keyword_', 'phase'] # i.e. groups
dict_of_terms = {}
for p in prefixes:
    # Regex to check if prefix is at the begining of a string (i.e. column name)
    re_search_str = r'^{}'.format(p)

    # establish empty list of terms associated with this prefix/group
    dict_of_terms[p] = []

    # look at each column name
    for n in column_info.index.tolist():

        # If column matches the prefix, add its human-readable name to the list
        # of terms
        if re.search(re_search_str, n) is not None:
            dict_of_terms[p].append(column_info.loc[n,'name'])

            # If this column is a term, indicate that in the column metadata
            column_info.loc[n,'isterm'] = True


# === MODEL: LINEAR REGRESSION + NORMALIZATION + TFORM Y + LASSO === #
reg = linear_model.Lasso(normalize=True)
reg.fit(X, ytform)
ypred = reg.predict(X)

# === PRINT OUTPUTS === #
print('\n ** Linear regression + normalization + transform y + LASSO')
print('Non-zero coefficients:')
for (c,f) in sorted(zip(reg.coef_, feature_names)):
    if abs(c) > 0:
        print('{:+0.2f}\t{}'.format(c, f))

print("RMS error: {:.2f}".format(mean_squared_error(ytform, ypred)**(1/2)))
print('Training R2 score: {:.2f}'.format(r2_score(ytform, ypred)))


# === MODEL: CV GRID SEARCH for hyperparameters === #
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

best_alpha = clf.best_params_['alpha']



# === MODEL: REFIT WITH BEST ALPHA === #
reg = linear_model.Lasso(alpha=best_alpha, normalize=True)
reg.fit(X, ytform)
ypred = reg.predict(X)

# === PRINT OUTPUTS === #
print('\n ** Linear regression + normalization + transform y + LASSO')
print('Non-zero coefficients:')
for (c,f) in sorted(zip(reg.coef_, feature_names)):
    if abs(c) > 1e-4:
        print('{:+0.2f}\t{}'.format(c, f))
print("RMS error: {:.2f}".format(mean_squared_error(ytform, ypred)**(1/2)))
print('Training R2 score: {:.2f}'.format(r2_score(ytform, ypred)))



# === MODEL: K-FOLD CROSS VALIDATION (i.e. check generalizable) === #
nfolds = 10
CV_scores = cross_val_score(reg, X, ytform,
                            cv=nfolds, 
                            scoring=make_scorer(r2_score))
print('{:d}-fold CV R2 score: {:0.2f}+/-{:0.2f}'
      .format(nfolds, CV_scores.mean(), CV_scores.std()))



# === MODEL: LEARNING CURVES === #

train_sizes, train_scores, test_scores = \
    learning_curve(reg, X, ytform, 
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

plt.ylabel('R2')
plt.xlabel('Training examples')
plt.ylim((-0.5, 1))
plt.legend(loc="best")
plt.show()


# === PLOT RESIDUALS === #

# hist of residuals
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(ytform - ypred, bins=50, orientation='horizontal')
plt.ylabel('residual')

# residuals vs predicted
plt.subplot(1, 2, 2)
plt.plot(ypred, ytform - ypred, '.')
plt.xlabel('predicted')
plt.ylabel('residual')
plt.show()

# === SAVE DATA & MODEL & METADATA VIA PICKEL === #

# DATA
filename = 'app_model1/Xraw_model1.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(Xraw, output_file)

filename = 'app_model1/yraw_model1.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(yraw, output_file)

filename = 'app_model1/X_model1.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(X, output_file)

filename = 'app_model1/y_model1.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(y, output_file)

# MODEL
filename = 'app_model1/reg_model1.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(reg, output_file)

# METADATA
filename = 'app_model1/human_names.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(human_names, output_file)

filename = 'app_model1/column_info.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(column_info, output_file)

filename = 'app_model1/dict_of_terms.pkl'
with open(filename, 'wb') as output_file:
    pk.dump(dict_of_terms, output_file)
