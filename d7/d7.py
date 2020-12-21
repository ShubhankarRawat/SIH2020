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

df = pd.read_csv('energydata.csv')
del df['date']

a = df.iloc[:500, :].values
a = pd.DataFrame(a)
a.columns = df.columns

col = list(a['Appliances'])
del a['Appliances']
a['Appliances'] = col

a.to_csv('d7.csv')

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

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


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




########################
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
import csv
from sklearn.metrics import confusion_matrix

import warnings


y_true = y_test



warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

def hyperopt(param_space, X_train, y_train, X_test, y_test, num_eval):
    d_train = lgb.Dataset(X_train, label = y_train)
    
    start = time.time()
    
    def objective_function(params):
        global y_true
        global s0
        global s1
        reg = lgb.LGBMRegressor(**params)
        reg.fit(X_train,y_train)
        #Prediction
        y_pred = reg.predict(X_test)
        
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        score = mean_squared_error(y_true, y_pred)
        
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function, 
                      param_space, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials,
                      rstate= np.random.RandomState(1))
#    loss = [x['result']['loss'] for x in trials.trials]
        
    if best_param['boosting_type'] == 0:
        best_param['boosting_type'] = 'gbdt'
    else:
        best_param['boosting_type']= 'dart'
    
    best_param['num_leaves'] = int(best_param['num_leaves'])
    best_param['max_depth'] = int(best_param['max_depth'])
    best_param['num_iterations'] = int(best_param['num_iterations'])
    best_param['min_child_weight'] = int(best_param['min_child_weight'])
    
    reg = lgb.LGBMRegressor(**best_param)
    reg.fit(X_train, y_train)
    #Prediction
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_true, y_pred)
    
    print("")
    print("##### ---->> Results (LightGBM) <<---- #####")
    print("MSE for test set: {}% ".format(mse) )
    print('\n')
#    print("Best parameters: ", best_param)
    print("Time elapsed: ", time.time() - start)
    print("Parameter combinations evaluated: ", num_eval)
    print('\n\n')
    
    
    return trials,best_param,mse


#p = {'boosting_type': 'gbdt' , 
#     'num_iterations' : 85 , 
#     'learning_rate': 0.1 ,
#     'num_leaves': 31,
#     'tree' : 'feature', 
#     'feature_fraction': 0.27, 
#     'bagging_fraction': 0.56435, 
#     'max_depth': 58,
#     'lambda_l1': 0.4898,
#     'lambda_l2': 1.954546, 
#     'min_split_gain': 0.00001235, 
#     'min_child_weight': 54}

param = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(2)),
    'max_depth': scope.int(hp.quniform('max_depth', 10, 1000, 1)),
    'num_iterations': scope.int(hp.quniform('num_iterations', 35, 1000, 1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 10, 500, 1)),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'feature_fraction': hp.uniform('feature_fraction', 0.0001, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.0001, 1.0),
    'lambda_l1': hp.uniform('lambda_l1', 0.000, 4.99999),
    'lambda_l2': hp.uniform('lambda_l2', 0.000, 6.99999),
    'min_split_gain': hp.loguniform('min_split_gain', np.log(0.000001), np.log(0.001)),
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 1e-5, 1e4, 10)),
    'scale_pos_weight': scope.int(hp.quniform('scale_pos_weight', 7, 40, 1)),
    'metric': 'binary_error'
}

results,params,acc = hyperopt(param, X_train, y_train, X_test, y_test, num_eval = 50)
with open('parameter.csv', 'a') as f:
    f.write("Accuracy : {0:.2f}% \n".format(acc))
    for key in params.keys():
        f.write("%s,%s\n"%(key,params[key]))
    f.write("\n")
        

reg = lgb.LGBMRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred_train = reg.predict(X_train)
Table(y_pred, y_pred_train, y_train, y_test, 'RFR', X_train, X_test)    
plotting(y_pred, 'RFR')



import pickle
pickle.dump(reg, open('d7_lightgbm.pkl','wb'))


