import numpy as np
import pandas as pd
import os
from scipy.sparse import coo_matrix
from sklearn.cross_validation import train_test_split

properties = ['PropertyTypeID',
              'DealTypeID',
              'HasCableTV',
              'HasCellar',
              'HasConservatory',
              'HasDishwasher',
              'HasDryer',
              'HasFireplace',
              'HasGarden',
              'HasGardenhouse',
              'HasLift',
              'HasPlayground',
              'HasTinkerroom',
              'HasWashingmachine',
              'HasWoodStove',
              'IsAttic',
              'IsBrightly',
              'IsBuildingLandConnected',
              'IsCentral',
              'IsChildFriendly',
              'IsCornerHouse',
              'IsFirstTimeUse',
              'IsFlatShare',
              'IsFurnished',
              'IsMiddleHouse',
              'IsMinergieCertified',
              'IsNewBuilding',
              'IsOldBuilding',
              'IsQuiet',
              'IsRaisedGroundFloor',
              'IsSunny',
              'IsUnderBuildingLaws',
              'IsUnderRoof',
              'IsWheelchairAccessable',
              'NumApartments',
              'NumBalconies',
              'NumBaths',
              'NumBathtubs',
              'NumFloors',
              'NumParkingIndoor',
              'NumParkingOutdoor',
              'NumPorches',
              'NumRooms',
              'NumShowers',
              'NumSwimmingPools',
              'NumTerraces',
              'NumToilets',
              'CountryID',
              'Zip',
              'Floor',
              'LivingSpace',
              'BalconyMeters',
              'GardenMeters',
              'GrossRent',
              'NetRent',
              'SideCost',
              'PurchasePrice',
              'PetsAllowed',
              'RenovationYear',
              'PriceModel',
              'GeoPosLng',
              'GeoPosLat',
              'ComparisPoints',
              'ComparisPrice',
              'UsefulArea']


class DataClass(object):
    def __init__(self,
                 files=("ads.csv", "users.csv", "ratings.csv"),
                 path='\\csv'):
        self.xFile_ = files[0]
        self.thetaFile_ = files[1]
        self.yFile_ = files[2]

        self.delimiter = ";"

        os.chdir(os.path.dirname(__file__) + '\\..' + path)
        self.csvPath = os.getcwd()

    def load_data(self):
        X = self.load_x()
        Theta = self.load_theta()
        y = self.load_y()

        return X, Theta, y

    def load_x(self, columnsX=properties):
        df = pd.read_csv(os.path.join(self.csvPath, self.xFile_), encoding="utf-8-sig", delimiter=self.delimiter)
        matrix = (np.array(df[['AdID']]), np.array(df[columnsX]))

        return matrix

    def load_theta(self):
        df = pd.read_csv(os.path.join(self.csvPath, self.thetaFile_), encoding="utf-8-sig", delimiter=self.delimiter)
        matrix = np.array(df['UserId'])

        return matrix

    def load_y(self):
        df = pd.read_csv(os.path.join(self.csvPath, self.yFile_), encoding="utf-8-sig", delimiter=self.delimiter)
        matrix = (np.array(df['RatingId']), np.array(df[['UserId', 'AdID', 'Rating']]))
        # users = np.array(df[['uIndex']]).flatten()
        # ads = np.array(df[['aIndex']]).flatten()
        # ratings = np.array(df[['Rating']]).flatten()
        #
        # rating_matrix = coo_matrix((ratings, (ads, users)), dims, dtype=np.float).toarray()
        # r = coo_matrix((ratings > 0, (ads, users)), dims, dtype=np.float).toarray()
        return matrix

    def init_random(self, len, width):
        new = np.random.rand(len, width)
        return new

    def init_y(self, y_raw, adIds, userIds, len, width):
        ratings = y_raw[:, 2]

        adIndexes = self.get_indexes(adIds, y_raw[:, 1])
        userIndexes = self.get_indexes(userIds, y_raw[:, 0])

        y = coo_matrix((ratings, (adIndexes, userIndexes)), (len, width), dtype=np.float).toarray()
        return y

    def get_R(self, y):
        return np.array(y > 0, dtype=np.int32)

    def get_indexes(self, allIds, idsToFind):
        unique = np.unique(allIds)
        unique.sort()
        unique = unique.tolist()

        idsToFind = np.array(idsToFind).astype(np.int32)

        indexes = []
        for i in range(idsToFind.size):
            indexes.append(unique.index(idsToFind[i]))

        return indexes

    def split_data(self, X, y, R, test_size=.25, random_state=42):
        '''
        :param X: Data set 1
        :param y: Data set 2
        :param test_size: represent the proportion of the dataset to include in the test split
        :param random_state: pseudo-random number generator state used for random sampling
        :return: X_train, X_test, y_train, y_test
        '''

        return train_test_split(X, y, R, test_size=test_size, random_state=random_state)
