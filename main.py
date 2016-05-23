# import Data.utilities as d
import CF_Hybrid.LowRankMatrixFactorization as h
import numpy as np
from numpy import *

import data.utilities as d

utilities = d.DataClass()

# load CSV files into matrices
X, ratings, hasRated = utilities.load_data()

# Normalize data
Xstd = (X - X.mean(axis=0)) / X.std(axis=0)
ystd = (ratings - ratings.mean(axis=0)) / ratings.std(axis=0)

# Set NaN values to zero
Xstd[isnan(Xstd)] = 0
ystd[isnan(ystd)] = 0

# Cross-Validation split
X_train, X_test, ratings_train, ratings_test, hasRated_train, hasRated_test = utilities.split_data(Xstd, ratings,
                                                                                                   hasRated)
# Generate theta
theta_train = np.random.rand(ratings_train.shape[1], X_train.shape[1])
theta_test = np.random.rand(ratings_test.shape[1], X_test.shape[1])

p = r_[X_train.T.flatten(), theta_train.T.flatten()]
num_x = X_train.shape[0]
num_users = theta_train.shape[0]
num_features = X_train.shape[1]

# Fit
estimator = h.EstimatorClass(X_train).fit(p, ratings_train, hasRated_train, num_x, num_users, num_features)

# Scores
print(estimator.score(theta_train, X_train, ratings_train))
print(estimator.score(theta_test, X_test, ratings_test))
print(estimator.score(theta_test, X_test, ratings_test, 'mean_squared_error'))
