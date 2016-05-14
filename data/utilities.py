import pandas as pd
from sklearn.cross_validation import train_test_split

class DataClass(object):

    def __init__(self, directory='file.csv'):
        '''
        :param directory: directory which leads to the csv file
        '''

        self.directory_ = directory

    def load_data(self):
        '''
        :return: dataset from the directory
        '''

        return pd.read_csv(self.directory_)

    def split_data(self, X, y, test_size=.25, random_state=42):
        '''
        :param X: Data set 1
        :param y: Data set 2
        :param test_size: epresent the proportion of the dataset to include in the test split
        :param random_state: pseudo-random number generator state used for random sampling
        :return: X_train, X_test, y_train, y_test
        '''

        return train_test_split(X, y, test_size, random_state)

