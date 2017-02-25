import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from theano.sandbox.cuda.basic_ops import row

def load_data(filepath='data/', split='train'):
	"""load all data"""
	if split == 'train':
		input_train_file = filepath + 'input_train.csv'
		output_train_file = filepath + 'challenge_output_data_training_file_oze_energies_optimizing_energy_consumptions.csv'
		input_train = pd.read_csv(input_train_file, delimiter=';', header=0, index_col=0, na_values=[''])
		output_train = pd.read_csv(output_train_file, delimiter=';', header=0, index_col=0, na_values=[''])
		return input_train, output_train
	else:
		input_test_file = filepath + 'input_test.csv'
		input_test = pd.read_csv(input_test_file, delimiter=';', header=0, index_col=0, na_values=[''])
		return input_test

def split_data(x, y=None, cut_percent=0.7):
	"""split training data to validate"""
	idx = int(x.shape[0]*cut_percent)
	if y is not None:
		return x[:idx], y[:idx], x[idx:], y[idx:]
	else:
		return x[:idx], x[idx:]

def get_bat(bat_id, x, y=None):
	"""get data for bat_id in {1,2,3,4}"""
	idx = x['Id_bat']==bat_id
	if y is not None:
		y = y[idx]
		return x[idx], y.loc[:, (y != 0.0).any(axis=0)]
	else:
		return x[idx]

def plot_data(x, y):
	"""plot train data"""
	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(2, sharex=True)
	x.plot(x='Time', y=['x1','x2', 'x3', 'x4', 'x5'], ax=axes[0])
	y.plot(x=x_train['Time'], y=['y1','y2','y3','y4','y5'], ax=axes[1])
	plt.show()


# x_train_all, y_train_all = load_data()

# x_train, y_train = get_bat(1, x_train_all, y_train_all)

# print y_train.ix[y_train.shape[0]-101:,:]

# x_train, y_train, x_test, y_test = split_data(x_train, y_train, 0.7)
# print x_train.mask()


# plot_data(x_train, y_train)

x_new_input_train, _ = load_data(split='train')
x_new_input_test = load_data(split='test')
x_new_input = pd.concat([x_new_input_train, x_new_input_test])


# bat_id = 1
x_train_all, y_train_all = load_data(split='train')
bat_id = 1
x_train, y_train = get_bat(bat_id, x_train_all, y_train_all)
x_test_all = load_data(split='test')
x_test = get_bat(bat_id, x_test_all)

x = pd.concat([x_train, x_test])
x = x.ix[:,'x1':]
x = np.ma.masked_invalid(x)

kl = KalmanFilter(n_dim_obs=5, n_dim_state=5)
model = kl.em(x, n_iter=40)
x_smooth_1, _ = model.smooth(x)

# bat_id = 2
x_train_all, y_train_all = load_data(split='train')
bat_id = 2
x_train, y_train = get_bat(bat_id, x_train_all, y_train_all)
x_test_all = load_data(split='test')
x_test = get_bat(bat_id, x_test_all)

x = pd.concat([x_train, x_test])
x = x.ix[:,'x1':]
x = np.ma.masked_invalid(x)

kl = KalmanFilter(n_dim_obs=5, n_dim_state=5)
model = kl.em(x, n_iter=40)
x_smooth_2, _ = model.smooth(x)

# bat_id = 3
x_train_all, y_train_all = load_data(split='train')
bat_id = 3
x_train, y_train = get_bat(bat_id, x_train_all, y_train_all)
x_test_all = load_data(split='test')
x_test = get_bat(bat_id, x_test_all)

x = pd.concat([x_train, x_test])
x = x.ix[:,'x1':]
x = np.ma.masked_invalid(x)

kl = KalmanFilter(n_dim_obs=5, n_dim_state=5)
model = kl.em(x, n_iter=40)
x_smooth_3, _ = model.smooth(x)

# bat_id = 4
x_train_all, y_train_all = load_data(split='train')
bat_id = 4
x_train, y_train = get_bat(bat_id, x_train_all, y_train_all)
x_test_all = load_data(split='test')
x_test = get_bat(bat_id, x_test_all)

x = pd.concat([x_train, x_test])
x = x.ix[:,'x1':]
x = np.ma.masked_invalid(x)

kl = KalmanFilter(n_dim_obs=5, n_dim_state=5)
model = kl.em(x, n_iter=40)
x_smooth_4, _ = model.smooth(x)

x_smooth = [x_smooth_1, x_smooth_2, x_smooth_3, x_smooth_4]

incr = [0, 0, 0, 0]
x_new_input_copy = x_new_input.copy()
for index, row in x_new_input_copy.iterrows():
	id_bat, time, x1, x2, x3, x4, x5 = row
	if np.isnan(x1):
		x_new_input.set_value(index, 'x1', x_smooth[id_bat - 1][incr[id_bat - 1], 0])
	if np.isnan(x2):
		x_new_input.set_value(index, 'x2', x_smooth[id_bat - 1][incr[id_bat - 1], 1])
	if np.isnan(x3):
		x_new_input.set_value(index, 'x3', x_smooth[id_bat - 1][incr[id_bat - 1], 2])
	if np.isnan(x4):
		x_new_input.set_value(index, 'x4', x_smooth[id_bat - 1][incr[id_bat - 1], 3])
	if np.isnan(x5):
		x_new_input.set_value(index, 'x5', x_smooth[id_bat - 1][incr[id_bat - 1], 4])
	
	incr[id_bat - 1] += 1

print x_new_input
	
x_new_input.to_csv('data/input_new.csv', sep=';')
