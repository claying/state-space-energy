import numpy as np
import pandas as pd

def load_data(filepath='data/', split='train'):
	"""load all data"""
	if split == 'train':
		input_train_file = filepath + 'input_train.csv'
		output_train_file = filepath + 'challenge_output_data_training_file_oze_energies_optimizing_energy_consumptions.csv'
		input_train = pd.read_csv(input_train_file, delimiter=';', header=0, index_col=0, na_values=[''])
		output_train = pd.read_csv(output_train_file, delimiter=';', header=0, index_col=0, na_values=[''])
		input_train['Time'] = pd.to_datetime(input_train['Time'])
		return input_train, output_train
	else:
		input_test_file = filepath + 'input_test.csv'
		input_test = pd.read_csv(input_test_file, delimiter=';', header=0, index_col=0, na_values=[''])
		input_test['Time'] = pd.to_datetime(input_test['Time'])
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
	# import matplotlib.dates as mdates
	fig, axes = plt.subplots(2, sharex=True)
	x.plot(x='Time', y=['x1','x2', 'x3', 'x4', 'x5'], ax=axes[0])
	y.plot(x=x['Time'], y=y.columns.get_values(), ax=axes[1])
	# fig.autofmt_xdate()
	# plt.draw()
	plt.show()

def rnn_load_data(bat_id, timesteps, cut_percent=0.8):
	u_train_all, y_train_all = load_data(filepath='data/complete/', split='train')
	u_train_val, y_train_val = get_bat(bat_id, u_train_all, y_train_all)
	u_train_val = u_train_val.ix[:,'x1':]
	# print u_train_val.shape
	# u_train_val = u_train_val.interpolate()
	x_train = np.zeros((y_train_val.shape[0] - timesteps, timesteps, y_train_val.shape[1]))
	y_train = np.zeros((y_train_val.shape[0] - timesteps, y_train_val.shape[1]))
	u_train = np.zeros((y_train_val.shape[0] - timesteps, timesteps, u_train_val.shape[1]))
	for i in range(y_train_val.shape[0] - timesteps):
		x_train[i, :, :] = y_train_val.iloc[i:i+timesteps, :]
		y_train[i, :] = y_train_val.iloc[i+timesteps, :]
		u_train[i, :, :] = u_train_val.iloc[i+1:i+timesteps+1, :]

	idx = int(u_train_val.shape[0]*cut_percent) - timesteps - 1
	# split data
	x_train, x_val = x_train[:idx], x_train[idx:]
	y_train, y_val = y_train[:idx], y_train[idx:]
	u_train, u_val = u_train[:idx], u_train[idx:]
	return dict(x=x_train, u=u_train, y=y_train), dict(x=x_val, u=u_val, y=y_val)

