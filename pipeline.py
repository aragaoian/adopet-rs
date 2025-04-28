from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from typing import List
import pandas as pd


class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)


class TransformationPipeline:
    def __init__(
        self,
        petsList: List[str],
        df,
        cols_to_drop: List[str],
        categorical_cols: List[str],
    ):
        self.df = df
        self.pets = petsList
        self.cols_to_drop = cols_to_drop
        self.categorical_cols = categorical_cols
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = MinMaxScaler()

    def oneHotEncodeAnimals(self, X):
        for animal in self.pets:
            X[animal] = X["animal"].apply(lambda x: 1 if animal in x else 0)
        X = X.drop(self.cols_to_drop, axis=1)[X.drop(self.cols_to_drop, axis=1).columns]
        return X

    def oneHotEncodeCols(self, X):
        self.ohe.fit(X)
        df_encoded = self.ohe.transform(X)  # arrumar aqui

        one_hot_df = pd.DataFrame(
            df_encoded,
            columns=self.ohe.get_feature_names_out(self.categorical_cols),
            index=X.index,
        )
        print(one_hot_df)

        X = pd.concat([X.drop(self.categorical_cols, axis=1), one_hot_df], axis=1)
        return X

    def pipeline(self):
        pipe = Pipeline(
            [
                (
                    "animals_encoding",
                    CustomFunctionTransformer(self.oneHotEncodeAnimals),
                ),
                ("cat_cols_encoding", CustomFunctionTransformer(self.oneHotEncodeCols)),
                ("scaler", self.scaler),
            ]
        )

        return pipe.fit_transform(self.df)
