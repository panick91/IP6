import numpy as np

class EstimatorClass(object):

    def __init__(self, fit_intercept=True):
        self.intercept_ = fit_intercept

    @staticmethod
    def costFunction(X, y, thetas, nlambda=.6):
        '''
        :param X: feature vector for items i
        :param y: ratings by user j on items i
        :param thetas: parameter vector for user j
        :param nlambda: regularization parameter
        :return: cost value
        '''

        lr = 1/2*(np.sum(np.square(X.dot(thetas)-y.size())))
        reg = nlambda/2*(np.square(X.dot(thetas)))

        return lr + reg

    @staticmethod
    def gradientFunction(X, y, thetas, alpha=.01, iters=1500):
        '''
        :param X: feature vector for items i
        :param y: ratings by user j on items i
        :param thetas: parameter vector for user j
        :param alpha: learning rate
        :param iters: number of iterations
        :return: updated thetas
        '''

        for i in np.arange(iters):
            thetas = thetas - alpha*(X.T.dot((X.dot(thetas)-y)))

        return thetas

    def fit(self):
        return "bla"

    def predict(self):
        return "bla"