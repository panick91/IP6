import numpy as np
from scipy.optimize import minimize

class EstimatorClass(object):

    def __init__(self, fit_intercept=True, threshold=0.5):
        self.fit_intercept_ = fit_intercept
        self.threshold_ = threshold

    @staticmethod
    def costFunction(thetas, X, y, nlambda=.6):
        '''
        :param thetas: parameter vector for user jfx
        :param X: feature vector for items i
        :param y: ratings by user j on items i
        :param nlambda: regularization parameter
        :return: cost value
        '''

        lr = 1/2*(np.sum(np.square(X.dot(thetas)-y)))
        reg = nlambda/2*(np.square(X.dot(thetas)))

        return lr + reg

    @staticmethod
    def gradientFunction(thetas, X, y, alpha=.01, nlambda=.6, iters=1500):
        '''
        :param thetas: parameter vector for user j
        :param X: feature vector for items i
        :param y: ratings by user j on items i
        :param alpha: learning rate
        :param iters: number of iterations
        :return: updated thetas
        '''

        for i in np.arange(iters):
            thetas = thetas - alpha*(1/y)*(X.dot(X.T.dot(thetas-y)))

        return thetas

    def fit(self, X, y, alpha=.01, iters=1500):
        '''
        :param X: feature vector for items i
        :param y: ratings by user j on items i
        :param alpha: learning rate
        :param iters: number of iterations
        :return: fitted model
        '''

        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.

        if self.fit_intercept_:
            x = np.insert(self.coef_, 0, self.intercept_)
            args = ((np.hstack((np.ones((X.shape[0], 1)), X)), y), alpha)
        else:
            x = self.coef_
            args = (X, y, alpha)

        result = minimize(
            EstimatorClass.costFunction,
            x,
            jac=EstimatorClass.gradientFunction,
            args=args,
            method='L-BFGS-B',
            options={'disp': True, 'maxiter': iters, })

        if self.fit_intercept_:
            self.intercept_ = result['x'][0]
            self.coef_ = result['x'][1:]
        else:
            self.coef_ = result['x']

        return self

    def predict(self, thetas, X):
        '''
        :param thetas: parameter vector for user j
        :param X: feature vector for items i
        :return: predicted model
        '''

        return X.dot(thetas)