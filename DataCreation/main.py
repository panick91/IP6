import numpy as np
import pandas as pd
import os


def get_array_from_csv(path, file, columns):
    os.chdir(os.path.dirname(__file__) + '\\..\\csv' + path)
    csv_path = os.getcwd()
    df = pd.read_csv(os.path.join(csv_path, file), encoding="utf-8-sig", delimiter=';')

    arr = np.array(df[columns])

    return arr


def export_csv(path, file, arr, header):
    os.chdir(os.path.dirname(__file__) + '\\..\\csv' + path)
    csv_path = os.getcwd()
    df = pd.DataFrame(arr)
    df.to_csv(os.path.join(csv_path, file),
                    header=header,
                    sep=';',
                    index=False,
                    encoding="utf-8-sig", )


X_id = get_array_from_csv('\\test', 'ads.csv', ['AdID'])
X_classes = np.random.randint(1, 11, X_id.shape)
X = np.hstack((X_id, X_classes))

Theta_id = get_array_from_csv('\\test', 'users.csv', ['UserId'])
Theta_classes = np.random.randint(1, 11, Theta_id.shape)
Theta = np.hstack((Theta_id, Theta_classes))

y = np.zeros((X.shape[0], Theta.shape[0]))

for i in range(Theta.shape[0]):
    t = Theta[i]
    cluster = np.where(X[:, 1] == t[1])[0]

    ratings = np.random.uniform(1., 4., y[:, i].shape)
    ratings[cluster] = np.random.uniform(3., 5., cluster.shape)

    y[:, i] = ratings

y_r = np.where(np.random.rand(y.shape[0], y.shape[1]) < 0.95, 0, 1)
y = y * y_r

y_coo = []
for index, x in np.ndenumerate(y):
    X_i = X[index[0], 0]
    T_i = Theta[index[1], 0]
    y_coo.append((index[0], X_i, index[1], T_i, x))

y_coo = np.array(y_coo)

export_csv('\\train', 'ratings.csv', y_coo, ['x_i', 'x_id', 't_i', 't_id', 'rating'])
export_csv('\\train', 'ads.csv', X, ['x_id', 'x_class'])
export_csv('\\train', 'users.csv', Theta, ['t_id', 't_class'])
