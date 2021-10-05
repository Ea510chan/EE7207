import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

class RBF(BaseEstimator, RegressorMixin):
    # define hidden neurons, gamma, weights
    def __init__(self, num_neurons=10, gamma=1 , w_weights=None):
        self.num_neurons = num_neurons
        self.gamma = gamma
        self.w_weights = w_weights

    def fit(self, x_train, y_train):
        x_train = np.c_[-1 * np.ones(x_train.shape[0]), x_train]

        kmeans = KMeans(n_clusters=self.num_neurons).fit(x_train)

        self.centers = kmeans.cluster_centers_

        H = rbf_kernel(x_train, self.centers, gamma=self.gamma)

        H = np.c_[-1 * np.ones(H.shape[0]), H]

        try:
            self.w_weights = np.linalg.lstsq(H, np.asmatrix(y_train).T, rcond=-1)[0]
        except:
            self.w_weights = np.linalg.pinv(H) @ y_train.reshape(-1, 1)
        return self

    def predict(self, x_test):
        x_test = np.c_[-1 * np.ones(x_test.shape[0]), x_test]

        H = rbf_kernel(x_test, self.centers, gamma=self.gamma)

        H = np.c_[-1 * np.ones(H.shape[0]), H]

        y_hat = np.asmatrix(H) @ np.asmatrix(self.w_weights)
        y_len = len(y_hat)
        for i in range(0, y_len):
            if y_hat[i,:] < 0:
                y_hat[i,:] = -1
            else:
                y_hat[i,:] = 1
        return y_hat

    def score(self, X, y, sample_weight=None):
        # from scipy.stats import pearsonr
        # r, p_value = pearsonr(y.reshape(-1, 1), self.predict(X))
        # return r ** 2
        return r2_score(y.reshape(-1, 1), self.predict(X))