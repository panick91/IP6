import Data.utilities as d
import CF_Hybrid.utilities as h
import numpy as np

data_import = d.DataClass('file.csv')
df = data_import.load_data()

X = np.array[[df['Cell1', 'Cell2', 'Cell3']]]
y = np.array[[df['JustAnotherCell']]]

X_train, X_test, y_train, y_test = data_import.split_data(X, y)

