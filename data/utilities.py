import numpy as np
import pandas as pd
import os
from scipy.sparse import coo_matrix
from sklearn.cross_validation import train_test_split


class DataClass(object):
    def __init__(self,
                 files=("ads.csv", "users.csv", "ratings.csv"),
                 columnsX=['PropertyTypeID', 'DealTypeID', 'GrossRent', 'NumRooms', 'LivingSpace']):
        self.xFile_ = files[0]
        self.thetaFile_ = files[1]
        self.yFile_ = files[2]

        self.columnsX = columnsX

        os.chdir(os.path.dirname(__file__) + '\\..\\csv')
        self.csvPath = os.getcwd()

    def load_data(self):

        X = self.load_xmatrix()
        thetas = self.load_thetamatrix()
        y = self.load_ymatrix((X.shape[0], thetas.shape[0]))

        return X, y

    def load_xmatrix(self):
        df = pd.read_csv(os.path.join(self.csvPath, self.xFile_), encoding="utf-8-sig", delimiter=';')
        matrix = np.array(df[self.columnsX])

        return matrix

    def load_thetamatrix(self):
        df = pd.read_csv(os.path.join(self.csvPath, self.thetaFile_), encoding="utf-8-sig", delimiter=';')
        matrix = np.random.rand(len(df), len(self.columnsX))

        return matrix

    def load_ymatrix(self, dims):
        df = pd.read_csv(os.path.join(self.csvPath, self.yFile_), encoding="utf-8-sig", delimiter=';')
        users = np.array(df[['uIndex']]).flatten()
        ads = np.array(df[['aIndex']]).flatten()
        ratings = np.array(df[['Rating']]).flatten()

        matrix = coo_matrix((ratings, (ads, users)), (817, 9253), dtype=np.float).toarray()
        return matrix

    def split_data(self, X, y, test_size=.25, random_state=42):
        '''
        :param X: Data set 1
        :param y: Data set 2
        :param test_size: represent the proportion of the dataset to include in the test split
        :param random_state: pseudo-random number generator state used for random sampling
        :return: X_train, X_test, y_train, y_test
        '''

        return train_test_split(X, y, test_size=test_size, random_state=random_state)
