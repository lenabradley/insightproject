"""
model1 - a module to fit a linear model to clinical trials data
=========================================================================

**model1** is a python module for fitting an ordinary linear regression model,
including feature scaling and LASSO (variable selection & regularization) via
scikit-learn
"""
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import data

first = False


# === GATHER DATA === #
if first:
    df = data.get_data(savename='rawdata.pkl', N=10,
                       dropna=True, fill_intelligent=True)
    [df, df_test] = data.split_data(df, save_suffix='data')

else:
    df = pd.read_pickle('training_data.pkl')
    df_test = pd.read_pickle('testing_data.pkl')

# Convert response and features to matrices
response_names = ['dropped', 'enrolled']
feature_names = []
for c in df.columns.tolist():
    if c not in response_names:
        feature_names.append(c)

X = df[feature_names].as_matrix()
y = (df['dropped']/df['enrolled']).as_matrix()

# === MODEL: LINEAR REGRESSION + NORMALIZATION + TFORM Y === #
ytform = y**(1/3)
reg = linear_model.LinearRegression(normalize=True)
reg.fit(X, ytform)
ypred = reg.predict(X)

# === PRINT OUTPUTS === #
print('\n ** Linear regression + normalization + transform y')
print('Coefficients:')
for (f,c) in zip(feature_names, reg.coef_):
    print('{:+0.2f}\t{}'.format(c,f))
print("Mean squared error: {:.2f}".format(mean_squared_error(ytform, ypred)))
print('Training Variance score: {:.2f}'.format(r2_score(ytform, ypred)))

# Plot residuals
plt.hist(ytform - ypred, bins=50)
plt.show()

# === MODEL: LINEAR REGRESSION + NORMALIZATION + TFORM Y + LASSO === #
reg = linear_model.Lasso(alpha=0.0001, normalize=True)
reg.fit(X, ytform)
ypred = reg.predict(X)

# === PRINT OUTPUTS === #
print('\n ** Linear regression + normalization + transform y + LASSO')
print('Non-zero coefficients:')
for (f,c) in zip(feature_names, reg.coef_):
    if abs(c)>0:
        print('{:+0.2f}\t{}'.format(c,f))
print("Mean squared error: {:.2f}".format(mean_squared_error(ytform, ypred)))
print('Training Variance score: {:.2f}'.format(r2_score(ytform, ypred)))

# Plot residuals
plt.hist(ytform - ypred, bins=50)
plt.show()

