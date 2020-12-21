# Hourly Energy Consumption Data

This dataset is considered to analyze and predict the hourly energy consumption of a building.
Estimating the energy consumption of a building in advance plays a crucial role in building energy management since, using this data the operator gets an idea of the future energy expenditure and can make certain arrangements to ensure it stays that way.
[link to dataset](https://www.kaggle.com/robikscube/hourly-energy-consumption)

The hourly energy energy consumption is predicted using a time series and supervised machine learning. A time series estimation problem can be converted to a supervised ML problem using some data transformation technique.

## Supervised machine learning-based hourly energy consumption prediction
  Time series modeling can be done via supervised ML techniques by pre-processing the dataset such that it becomes a regression (supervised ML) problem. In regression, an ML model is trained on a training dataset (input-output pairs), and then predictions are obtained for the test dataset (inputs only). A univariate time series prediction problem can be reduced to a regression problem by considering k-1 preceding data units as input and kth data unit as output for training. For instance, in a univariate time series with n number of data points, one can have the following input-output training pairs: (1, 2, …, k-1; k), (2, 3, …, k; k+1), …, (n-k, n-k+1, …, n-1; n) where k < n. The value of k was considered as 5 in this project. 

## Performance evaluation metrics
### Root mean square error
RMSE is a popular evaluation metric used to measure the square root of the average of squared errors. As the name suggests, RMSE is calculated by taking the square root of MSE. RMSE has the same unit as that of the data, and therefore, it is scale-dependent. Further, a low value of RMSE signifies that the model’s predictions are very close to the actual values, and a higher value signifies that the model is performing poorly. RMSE is calculated using the following equation.
             RMSE= √(1/m ∑_(i=1)^m▒(y_i-(y_i ) ̂ )^2 )                                            (4)
Where y_i, m, and (y_i ) ̂ represent the actual value, the total number of data instances, and predicted value, respectively.
3.3.3 Mean absolute error
MAE is used to measure the average magnitude of the difference between actual and predicted values. MAE has the same unit as that of the data, and hence, it is also scale-dependent like RMSE. Higher MAE values signify that the model’s predictions highly vary from the actual values, which is undesirable. The following equation is used for calculating MAE.
                                         MAE=  1/m ∑_(i=1)^m▒|y_i-(y_i ) ̂ |                                                 (5)
Where y_i, m, and (y_i ) ̂ represent the actual value, the total number of data instances, and predicted value, respectively.
3.3.4 Mean absolute percentage error
MAPE is a popular evaluation metric used for the evaluation of regression models. It is used to measure the average magnitude of the difference between actual and predicted values in percentage hence, increasing interpretability. Higher values of MAPE signify that the model is performing poorly. MAPE is calculated using the following equation.
MAPE=100*1/m ∑_(i=1)^m▒|(y_i-(y_i ) ̂)/y_i |                                             (6)
Where, m, y_i, and (y_i ) ̂ represent the total number of data instances, actual value, and predicted value, respectively.

