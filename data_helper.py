import pandas as pd

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