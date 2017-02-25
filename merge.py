import pandas as pd 


def merge(filepath='output/checkpoint_val/'):
	output = []
	for bat_id in range(1,5):
		df = pd.read_csv(filepath+'output'+str(bat_id)+'.csv',  delimiter=';', header=0, index_col=0)
		output.append(df)

	return pd.concat(output)

# output = merge('output_complete/')
# print output
# output.to_csv('output_complete/output.csv', sep=';')



from data_helper import load_data, split_data, get_bat
import numpy as np 
x_train_all, y_train_all = load_data(split='train')
bat_id = 1
x_train, y_train = get_bat(bat_id, x_train_all, y_train_all)
df = pd.read_csv('output_complete/output'+str(bat_id)+'.csv',  delimiter=';', header=0, index_col=0)



df = pd.concat((y_train, df))

# df.ix[:,'y2'] = 0.0
# df.to_csv('output_complete/output'+str(bat_id)+'.csv', sep=';')

import matplotlib.pyplot as plt
df.plot(x=np.arange(df.shape[0]))
plt.legend(loc='best')
plt.show()