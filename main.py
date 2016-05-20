# import Data.utilities as d
import CF_Hybrid.LowRankMatrixFactorization as h
import numpy as np

import data.utilities as d
utilities = d.DataClass()
X, y, R = utilities.load_data()

X_train, X_test, y_train, y_test, R_train, R_test = utilities.split_data(X, y, R)

theta_train = 5 * np.random.rand(y_train.shape[1], X_train.shape[1])
theta_test = 5 * np.random.rand(y_test.shape[1], X_test.shape[1])

# data_import = d.DataClass('/Users/trish/Desktop/Movie Ratings.csv')
# df = data_import.load_data()
#
# X = np.array(df[['UserId', 'MovieId']])
# y = np.array(df[['Rating']])
#
# X_train, X_test, y_train, y_test = data_import.split_data(X, y)
#
estimator = h.EstimatorClass().fit(X_train, y_train, theta_train, R_train)
print(estimator.predict(theta_train, X_train))
print(estimator.score(theta_train, X_train, y_train))

