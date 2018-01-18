import demo
import pandas as pd

# ============ GATHER DATA
df = pd.read_pickle('training_data.pkl')


# ============= FIT MODEL
res = demo.fit_model(df)

# Show results
print(res.summary())
demo.diagnotic_plots(res, show=True)

# Save linear model results via pickle (data included)
res.save('training_res.pkl', remove_data=False)








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
