from sklearn import metrics
import numpy as np


class Measures:

    @staticmethod
    def silhouette(x, y_pred):
        return (1 - metrics.silhouette_score(x, y_pred)) / 2

    @staticmethod
    def davies_bouldin(x, y_pred):
        return np.tanh(metrics.davies_bouldin_score(x, y_pred))

    @staticmethod
    def calinski_harabasz(x, y_pred):
        return 1 - np.tanh(metrics.calinski_harabasz_score(x, y_pred))
