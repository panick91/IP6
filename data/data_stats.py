# import Data.utilities as d
import numpy as np
from numpy import *

import data.utilities as d

utilities = d.DataClass(path='\\csv\\test')

# load CSV files into matrices
X, ratings, hasRated = utilities.load_data()

print('X shape: ' + str(X.shape))
print('ratings shape: ' + str(ratings.shape))

print('Number of ratings: ' + str(np.sum(hasRated)))

r = hasRated.sum(axis=0)
r[::-1].sort()
print('Users with most ratings:')
print(r[:10])

r = hasRated.sum(axis=1)
r[::-1].sort()
print('Ads with most ratings:')
print(r[:10])
