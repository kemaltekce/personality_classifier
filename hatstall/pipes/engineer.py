import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class Debugger(BaseEstimator, TransformerMixin):

    def transform(self, X, *_):
        import ipdb
        ipdb.set_trace()

    def fit(self, *_):
        return self


class PostsJoiner(BaseEstimator, TransformerMixin):

    def transform(self, X, *_):
        return [' '.join(x) for x in X]

    def fit(self, *_):
        return self


class AverageWordCalculator(BaseEstimator, TransformerMixin):

    def transform(self, X, *_):
        return [[np.mean([len(x.split()) for x in X_])] for X_ in X]

    def fit(self, *_):
        return self
