import matplotlib.pyplot as plt
import numpy as np 
from data_helper import load_data, split_data, get_bat
from public_mean_square_error import score_function
import period

x_train_all, y_train_all = load_data()
bat_id = 1
x_train, y_train = get_bat(bat_id, x_train_all, y_train_all)
T = y_train.shape[0]
percent = 0.8

x_train, y_train, x_test, y_test = split_data(x_train, y_train, percent)

regimes_array = period.regimes(T, t0=0,di=(8,19), ni=(21,6), tw0=24, wl=48)

# print regimes_array[:100]

# x_train = x_train.ix[:,'x1':].as_matrix()
# y_train = y_train.as_matrix()
# day_period = regimes_array == 0

# print day_period[:200]
# print y_train[day_period,:][:200]
# plt.plot(y_train[day_period,:])
# # plt.plot(y_train[1-day_period,:])
# plt.show()

regimes_train, regimes_test = split_data(regimes_array, cut_percent=percent)

day_train = regimes_train==0
day_test = regimes_test==0


x_train = x_train.ix[:,'x1':]
x_test = x_test.ix[:,'x1':]

x_train = x_train.ix[:, [0,2,3]]
x_test = x_test.ix[:, [0,2,3]]

x_train = np.ma.masked_invalid(x_train)
x_test = np.ma.masked_invalid(x_test)
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()

x_train = x_train[day_train,:]
y_train = y_train[day_train,:]
x_test = x_test[day_test, :]
y_test = y_test[day_test, :]

x_dim = x_train.shape[1]
y_dim = y_train.shape[1]
state_dim = y_dim


from pykalman import VariantKalmanFilter

kl = VariantKalmanFilter(observation_matrices=np.eye(y_dim, state_dim), transition_offsets=np.zeros((state_dim, x_dim))
	, n_dim_obs=y_dim, n_dim_state=state_dim, n_dim_control=x_dim)


em_vars = ['transition_matrices', 'transition_covariance', 'observation_offsets', 'observation_covariance', 'initial_state_mean', 'observation_matrices']

model = kl.em(y_train, x_train, em_vars=em_vars, n_iter=40)


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

y_pred, pred_upper, pred_lower = model.forcast(np.vstack((x_train[-1,:], x_test)), x0, P0)

print(y_pred)
y_pred = np.maximum(y_pred, 0)

print('total error: %f', score_function(y_test, y_pred))


plt.plot(y_pred, label='pred')
plt.plot(np.arange(y_test.shape[0]), y_test, label='obs')
colors = ['b', 'g', 'r', 'c', 'm']
for i in range(y_dim):
	plt.fill_between(np.arange(y_test.shape[0]), pred_lower[:,i], pred_upper[:,i], color=colors[i], alpha=0.2, label='pred')
plt.grid(True)
plt.legend(loc='best', prop={'size':8})
plt.show()

