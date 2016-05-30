import numpy as np
import scipy
from numpy import r_
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class EstimatorClass(object):
    def __init__(self, X):

        self.X = X
        self.theta = np.array(0)

    @staticmethod
    def unroll_params(p, num_x, num_users, num_features):
        # Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
        X = p[:num_x * num_features]
        X = X.reshape((num_features, num_x)).transpose()
        theta = p[num_x * num_features:]
        theta = theta.reshape(num_features, num_users).transpose()
        return X, theta

    @staticmethod
    def costFunction(p, y, r, num_x, num_users, num_features, alpha):

        X, theta = EstimatorClass.unroll_params(p, num_x, num_users, num_features)
        cost = np.sum((X.dot(theta.T) * r - y) ** 2) / 2
        reg = (alpha / 2) * (np.sum(theta ** 2) + np.sum(X ** 2))
        return cost + reg

    @staticmethod
    def gradientFunction(p, y, r, num_x, num_users, num_features, alpha):

        X, theta = EstimatorClass.unroll_params(p, num_x, num_users, num_features)
        difference = X.dot(theta.T) * r - y
        X_grad = difference.dot(theta) + alpha * X
        theta_grad = difference.T.dot(X) + alpha * theta
        return r_[X_grad.T.flatten(), theta_grad.T.flatten()]

    def fit(self, p, y, R, num_x, num_users, num_features, alpha=30):

        min = scipy.optimize.fmin_cg(EstimatorClass.costFunction, fprime=EstimatorClass.gradientFunction, x0=p,
                                     args=(y, R, num_x, num_users, num_features, alpha),
                                     maxiter=1500, disp=True, full_output=True)

        cost, new_p = min[1], min[0]

        X, theta = EstimatorClass.unroll_params(new_p, num_x, num_users, num_features)
        self.X = X
        self.theta = theta

        return self

    def predict(self, thetas, X):

        return X.dot(thetas.T)

    def score(self, thetas, X, y, method='mean_squared_error'):

        y_pred = self.predict(self.theta, self.X)

        if method == 'R2-Score':
            return r2_score(y, y_pred, multioutput='uniform_average')
        if method == 'mean_squared_error':
            return mean_squared_error(y, y_pred)
