import numpy as np 
from data_helper import load_data, split_data, get_bat
from public_mean_square_error import score_function
import period
import pandas as pd 

x_train_all, y_train_all = load_data(split='train')
bat_id = 2
x_train, y_train = get_bat(bat_id, x_train_all, y_train_all)
x_test_all = load_data(split='test')
x_test = get_bat(bat_id, x_test_all)

T_train = x_train.shape[0]
T_test = x_test.shape[0]

T = T_train + T_test

y_pred_pd = pd.DataFrame(np.zeros((x_test.shape[0], y_train_all.shape[1])), index=x_test.index.values, columns=y_train_all.columns.values)
y_pred_pd.index.name = y_train.index.name
col_names = y_train.columns.values
# y_test.ix[:, y_train.columns.values] = np.ones((x_test.shape[0], 4))

regimes_array = period.regimes(T, t0=0,di=(8,19), ni=(21,6), tw0=24, wl=48)
regimes_train = regimes_array[:T_train]
regimes_test = regimes_array[T_train:]
# regimes_train, regimes_test = split_data(regimes_array, cut_percent=percent)


x_train = x_train.ix[:,'x1':]
x_test = x_test.ix[:,'x1':]

x_train = x_train.ix[:, [0,2,3]]
x_test = x_test.ix[:, [0,2,3]]

x_train = np.ma.masked_invalid(x_train)
x_test = np.ma.masked_invalid(x_test)
y_train = y_train.as_matrix()



x_dim = x_train.shape[1]
y_dim = y_train.shape[1]
state_dim = y_dim


from pykalman import MultiRegimes

kl = MultiRegimes(observation_matrices=np.eye(y_dim, state_dim), transition_offsets=np.zeros((5,state_dim, x_dim)),
	regimes_array=regimes_train
	, n_dim_obs=y_dim, n_dim_state=state_dim, n_dim_control=x_dim, n_dim_regimes=5)


em_vars = ['transition_matrices', 'transition_covariance', 'observation_offsets', 'observation_covariance', 'initial_state_mean']

model = kl.em(y_train, x_train, em_vars=em_vars, n_iter=80)


A = model.transition_matrices
B = model.transition_offsets
sigma = model.transition_covariance
C = model.observation_matrices
D = model.observation_offsets
eta = model.observation_covariance
print(A)
print(B)
print(sigma)
print(C)
print(D)
print(eta)

Ks = model.smooth(y_train, x_train)
x = Ks[0]

x0 = Ks[0][-1]
P0 = Ks[1][-1]

y_pred, pred_upper, pred_lower = model.forcasting(np.vstack((x_train[-1,:], x_test)), np.hstack((regimes_train[-1], regimes_test)), x0, P0)
y_pred = np.maximum(y_pred, 0)

y_pred_pd.ix[:, col_names] = y_pred
y_pred_pd.to_csv('output/output'+str(bat_id)+'.csv', sep=';')


import matplotlib.pyplot as plt

plt.plot(np.vstack((y_train, y_pred)))
colors = ['b', 'g', 'r', 'c', 'm']
for i in range(y_dim):
	plt.fill_between(np.arange(T_train, T), pred_lower[:,i], pred_upper[:,i], color=colors[i], alpha=0.2, label='pred')
plt.grid(True)
plt.show()





