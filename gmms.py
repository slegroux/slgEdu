import operator
import numpy as np
from sklearn.mixture import GMM
from IPython import embed


class GMMS(object):
    def __init__(self, n_components=32):
        self.gmms = []
        self.n_components = n_components
        self.y = []

    def fit_new(self, x, label):
        self.y.append(label)
        gmm = GMM(self.n_components)
        gmm.fit(x)
        self.gmms.append(gmm)

    def gmm_score(self, gmm, x):
        return np.sum(gmm.score(x))

    def predict_one(self, x):
        # embed()
        scores = [self.gmm_score(gmm, x) / len(x) for gmm in self.gmms]
        result = [(self.y[index], value) for (index, value) in enumerate(scores)]
        p = max(result, key=operator.itemgetter(1))
        return p[0]