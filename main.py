import Data.utilities as d
import CF_Hybrid.utilities as h
import numpy as np

data_import = d.DataClass('/Users/trish/Desktop/Movie Ratings.csv')
df = data_import.load_data()

X = np.array(df[['UserId', 'MovieId']])
y = np.array(df[['Rating']])

X_train, X_test, y_train, y_test = data_import.split_data(X, y)

estimator = h.EstimatorClass().fit(X_train, y_train)

