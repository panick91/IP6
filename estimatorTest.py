# import Data.utilities as d
import CF_Hybrid.LowRankMatrixFactorization as h
from sklearn.metrics import mean_squared_error
from scipy.optimize import check_grad
import numpy as np
from numpy import r_

import data.utilities as d

utilities = d.DataClass()
X, y, R = utilities.load_data()

Xstd = (X - X.mean(axis=0)) / X.std(axis=0)
ystd = (y - y.mean(axis=0)) / y.std(axis=0)

X_train, X_test, y_train, y_test, R_train, R_test = utilities.split_data(Xstd, y, R)

theta_train = np.random.rand(y_train.shape[1], X_train.shape[1])
theta_test = np.random.rand(y_test.shape[1], X_test.shape[1])

p = r_[X_train.T.flatten(), theta_train.T.flatten()]
num_x = X_train.shape[0]
num_users = theta_train.shape[0]
num_features = X_train.shape[1]
args = (np.hstack((np.ones((X_train.shape[0], 1)), X_train)), y_train, R_train, .01)
check_grad(h.EstimatorClass.costFunction,
           h.EstimatorClass.gradientFunction,
           p, y_train, R_train,
           num_x, num_users, num_features, 30)

estimator = h.EstimatorClass(X_train).fit(p, y_train, R_train, num_x, num_users, num_features)
print(estimator.score(theta_train, X_train, y_train))
print(estimator.score(theta_test, X_test, y_test))
print(mean_squared_error(y_test, estimator.predict(theta_test, X_test)))
