import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score


class EstimatorClass(object):
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: fit model with an intercept
        '''

        self.fit_intercept_ = fit_intercept

    @staticmethod
    def costFunction(thetas, X, y, R, nlambda=.6):
        '''
        :param thetas: parameter vector for user jfx
        :param X: feature vector for items i
        :param y: ratings by user j on items i
        :param nlambda: regularization parameter
        :return: cost value
        '''
        thetas = thetas.reshape(y.shape[1], X.shape[1])
        lr = 1 / 2 * (np.sum(np.square((X.dot(thetas.T) - y) * R)))
        reg = nlambda / 2 * np.sum(np.square(thetas[:, 1:X.shape[1]]))

        return lr + reg

    @staticmethod
    def gradientFunction(thetas, X, y, R, alpha=.01, lamb=0.8):
        '''
        :param thetas: parameter vector for user j
        :param X: feature vector for items i
        :param y: ratings by user j on items i
        :param alpha: learning rate
        :param iters: number of iterations
        :return: updated thetas
        '''

        thetas = thetas.reshape(y.shape[1], X.shape[1])
        # theta0 = alpha * np.sum(((X.dot(thetas.T) - y) * R).T.dot(X[:, 0].reshape(X.shape[0], 1)), axis=1)

        # grad = [alpha * np.sum((((X.dot(thetas.T) - y) * R).T.dot(X[:, j + 1].reshape(X.shape[0], 1)) + lamb * (
        #     thetas[:, j + 1].reshape(thetas.shape[0], 1))), axis=1) for j in range(X.shape[1] - 1)]

        theta_grad = ((X.dot(thetas.T) - y) * R).T.dot(X)
        theta_grad += lamb * thetas

        return theta_grad.ravel()

    def fit(self, X, y, thetas, R, alpha=.01, iters=1500):
        '''
        :param X: feature vector for items i
        :param y: ratings by user j on items i
        :param alpha: learning rate
        :param iters: number of iterations
        :return: fitted model
        '''

        self.coef_ = thetas
        self.intercept_ = 0.

        if self.fit_intercept_:
            thetas = np.hstack((np.ones((self.coef_.shape[0], 1)), self.coef_))
            args = (np.hstack((np.ones((X.shape[0], 1)), X)), y, R, alpha)
        else:
            thetas = self.coef_
            args = (X, y, R, alpha)

        result = minimize(
            EstimatorClass.costFunction,
            thetas,
            jac=EstimatorClass.gradientFunction,
            args=args,
            method='L-BFGS-B',
            options={'disp': True, 'maxiter': iters,})

        if self.fit_intercept_:
            self.intercept_ = result['x'][0]
            self.coef_ = result['x'][1:]
        else:
            self.coef_ = result['x']

        return self

    def predict(self, thetas, X):
        '''
        :param thetas: parameter vector for user j
        :param X: samples
        :return: predicted values
        '''

        return X.dot(thetas.T)

    def score(self, thetas, X, y):
        '''
        :param X: test samples
        :param y: true values for X
        :return: coefficient of determination R^2 of the prediction
        '''

        y_pred = self.predict(thetas, X)

        return r2_score(y, y_pred, multioutput='variance_weighted')
