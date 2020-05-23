from sklearn.base import BaseEstimator, TransformerMixin


class PostsJoiner(BaseEstimator, TransformerMixin):

    def transform(self, X, *_):
        return [' '.join(x) for x in X]

    def fit(self, *_):
        return self
