from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from typing import List
from utils import CustomFunctionTransformer


class TransformationPipeline:
    def __init__(
        self,
        petsList: List[str],
        cols_to_drop: List[str],
        categorical_cols: List[str],
    ):
        self.pets = petsList
        self.cols_to_drop = cols_to_drop
        self.categorical_cols = categorical_cols
        self.encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            categories=[
                ["small", "medium", "big"],
                ["yes", "no"],
                ["200-499", "500-749", "750-999", "1000+"],
                ["yes", "no"],
            ],
        )
        self.scaler = MinMaxScaler()

    def oneHotEncodeAnimals(self, X):
        for animal in self.pets:
            X[animal] = X["animal"].apply(lambda x: 1.0 if animal in x else 0.0)
        X = X.drop(self.cols_to_drop, axis=1)[X.drop(self.cols_to_drop, axis=1).columns]
        return X

    def oneHotEncodeCols(self, X):
        self.encoder.fit(X[self.categorical_cols])
        df_encoded = self.encoder.transform(X[self.categorical_cols])  # arrumar aqui

        one_hot_df = pd.DataFrame(
            df_encoded,
            columns=self.encoder.get_feature_names_out(self.categorical_cols),
            index=X.index,
        )

        X = pd.concat([X.drop(self.categorical_cols, axis=1), one_hot_df], axis=1)
        return X

    def finalTransform(self, X):
        if "id" in X.columns:
            X = X.drop("id", axis=1)
        if X.shape[1] > 16:
            X = X.iloc[:, :16]
        return X

    def pipeline(self):
        return Pipeline(
            [
                (
                    "animals_encoding",
                    CustomFunctionTransformer(self.oneHotEncodeAnimals),
                ),
                ("cat_cols_encoding", CustomFunctionTransformer(self.oneHotEncodeCols)),
                ("final_transform", CustomFunctionTransformer(self.finalTransform)),
            ]
        )
