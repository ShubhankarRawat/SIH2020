import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
from sklearn.metrics import r2_score
from prettytable import PrettyTable
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
import os

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


df = pd.read_csv('AEP_hourly.csv')
energy_series = df.iloc[:, -1].values
# define input sequence
train_series = energy_series[:int(0.8*len(energy_series))]
test_series = energy_series[int(0.8*len(energy_series)):]
# choose a number of time steps
n_steps = 5
# split into samples
X_train, y_train = split_sequence(train_series, n_steps)
X_test, y_test = split_sequence(test_series, n_steps)

# data = pd.DataFrame(X_train[:500, :])
# data['target'] = y_train[:500]

# data.to_csv('d2.csv', index = False)



def adj_r2(r_squared, yhat, y, X):
    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    return(adjusted_r_squared)

def mape(y_true, y_hat):
    s = 0
    for i in range(0, len(y_true)):
        b = y_hat[i] - y_true[i]
        if b < 0:
            b = b*(-1)
        s = s + (b/y_true[i])
    return(s/len(y_true))


def Table(y_pred, y_pred_train, y_train, y_test, name, X_train, X_test):
    x = PrettyTable()
    x.field_names = ['', '{}'.format(name), ' ']
    x.align = 'c'

    x.add_row(["Metric", "Value", "Data"])    
    x.add_row(['-----------------------', '----------------------', '--------------'])
    x.add_row(["RMS", sqrt(mean_squared_error(y_test, y_pred)), 'Test'])
    x.add_row(["RMS", sqrt(mean_squared_error(y_train, y_pred_train)), 'Train'])
    x.add_row(['', '', ''])
    x.add_row(["R2", r2_score(y_test, y_pred), 'Test'])
    x.add_row(["R2", r2_score( y_train, y_pred_train), 'Train'])
    x.add_row(['', '', ''])
    x.add_row(['Adjusted R2', adj_r2(r2_score(y_test, y_pred), y_pred, y_test, X_test), 'Test'])
    x.add_row(['Adjusted R2', adj_r2(r2_score(y_train, y_pred_train), y_pred_train, y_train, X_train), 
                'Train'])
    x.add_row(['','',''])
    x.add_row(['Mean Absolute Error', mean_absolute_error(y_test, y_pred), 'Test'])
    x.add_row(['Mean Absolute Error', mean_absolute_error(y_train, y_pred_train), 'Train'])
    x.add_row(['','',''])
    x.add_row(['Explained Variance Score', explained_variance_score(y_test, y_pred), 'Test'])
    x.add_row(['Explained Variance Score', explained_variance_score(y_train, y_pred_train), 'Train'])
    x.add_row(['','',''])
    x.add_row(['MAPE', mape(y_test, y_pred), 'Test'])
    x.add_row(['MAPE', mape(y_train, y_pred_train), 'Train'])
    
    print(x)


from sklearn.linear_model import LinearRegression as LR
reg = LR()
reg.fit(X_train, y_train)
y_pred_lr = reg.predict(X_test)
y_pred_train_lr = reg.predict(X_train)
Table(y_pred_lr, y_pred_train_lr, y_train, y_test, 'LR', X_train, X_test)


####============= ANN
from keras.models import Sequential
from keras.layers import Dense
# define model
n_input = len(X_train)
n_nodes = 500
n_epochs = 100
n_batch = 8

model = Sequential()
model.add(Dense(2048, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=1)



######============================================================#########
########==================== LSTM =================================#########
######============================================================#########
from keras.layers import LSTM
import keras.backend as K
K.clear_session()
n_steps = 5

pX_train, py_train = X_train, y_train
pX_test, py_test = X_test, y_test

ptrain_x = pX_train.reshape((pX_train.shape[0], pX_train.shape[1], 1))
ptest_x = pX_test.reshape((pX_test.shape[0], pX_test.shape[1], 1))

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
# fit
history = model.fit(ptrain_x, py_train, epochs=1, batch_size = 32, verbose=1)


y_pred_lstm = model.predict(ptest_x)

y_pred_train_lstm = model.predict(ptrain_x)

# cum_y_pred = cumulative_cases(y_pred, cases)
# cum_y_pred_train = cumulative_cases(y_pred_train, cases)

# Table(cum_y_pred, cum_y_pred_train, y_train, y_test, 'LSTM', X_train, X_test)
# plotting_cases(cum_y_pred, y_test)

Table(y_pred_lstm, y_pred_train_lstm, y_train, y_test, 'LSTM', X_train, X_test)

import pickle
pickle.dump(reg, open('d2_lr.pkl','wb'))

#### how to predict for single row values
a = [213, 12314, 1234, 4521, 12313]
a = array(a)
a = a.reshape(1, -1)
ax = a.reshape((a.shape[0], a.shape[1], 1))
model.predict(ax)