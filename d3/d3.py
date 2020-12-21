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


df = pd.read_csv('electricity_france.csv')
del df['Date']

a = df.describe()

df.isnull().sum()
df.Voltage = df.fillna(df['Voltage'].value_counts().index[0])

a = df.iloc[:500, :].values
data = pd.DataFrame(a)
data.columns = df.columns

data.to_csv('d3.csv', index=None)

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
    return((s/len(y_true))*100)


def plotting(y_pred, name):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 51, 10)
    minor_ticks = np.arange(0, 51, 1)
    
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    ax.grid(which = 'both')
    
    # Or if you want different settings for the grids:
    ax.grid(which = 'minor', alpha=0.2)
    ax.grid(which = 'major', alpha=0.5)
    a = [i for i in  range(0, 50)]
    
    plt.plot(a, y_test[:50], color = 'b', label = 'Actual')
    plt.scatter(a, y_test[:50], color = 'b', label = 'Actual')
    plt.plot(a, y_pred[:50], color = 'r', label = 'Predicted')
    plt.scatter(a, y_pred[:50], color = 'r', label = 'Predicted')
    plt.legend()
    plt.title('First 50 values for {} for Test Set'.format(name))
    plt.show()

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

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


for i in range(0, len(y)):
    if y[i] == 0:
        y[i] = 1

from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression as LR
reg = LR()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred_train = reg.predict(X_train)
Table(y_pred, y_pred_train, y_train, y_test, 'LR', X_train, X_test)


from sklearn.tree import DecisionTreeRegressor as DTR
reg = DTR()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred_train = reg.predict(X_train)
Table(y_pred, y_pred_train, y_train, y_test, 'DTR', X_train, X_test)


from sklearn.ensemble import RandomForestRegressor as RF
reg = RF()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred_train = reg.predict(X_train)
Table(y_pred, y_pred_train, y_train, y_test, 'RFR', X_train, X_test)    
plotting(y_pred, 'RFR')



from lightgbm import LGBMRegressor as lgb
reg = lgb()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred_train = reg.predict(X_train)
Table(y_pred, y_pred_train, y_train, y_test, 'LGBM', X_train, X_test)    
plotting(y_pred, 'RFR')


from xgboost import XGBRegressor
reg = XGBRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred_train = reg.predict(X_train)
Table(y_pred, y_pred_train, y_train, y_test, 'LGBM', X_train, X_test)    
plotting(y_pred, 'RFR')


import pickle
pickle.dump(reg, open('d3_rf.pkl','wb'))


from sklearn.svm import SVR
reg = SVR()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred_train = reg.predict(X_train)
Table(y_pred, y_pred_train, y_train, y_test, 'LGBM', X_train, X_test)    
plotting(y_pred, 'RFR')

