from sklearn.base import BaseEstimator, TransformerMixin


class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)
