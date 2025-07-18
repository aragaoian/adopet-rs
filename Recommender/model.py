from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# TODO
# pesos dinamicos
# diminuição gradual dos pesos no calcFeaturesWeigths


class ContentBasedFiltering:
    def __init__(
        self,
        user_profile,
        items_profile,
        items_not_piped,
        weigths=None,
        test_weigths=False,
    ):
        self.user_profile = user_profile.copy()
        self.items_profile = (
            items_profile.copy()
        )  # evitar incosistência nas transformações
        self.items_not_piped = items_not_piped
        self.cs = 0
        if not test_weigths:
            self.weights = (
                (
                    np.array(
                        [
                            0.6,
                            0.6,
                            0.6,
                            0.6,
                            0.6,
                            1.0,
                            1.0,
                            1.0,
                            0.2,
                            0.2,
                            0.8,
                            0.8,
                            0.8,
                            0.8,
                            0.4,
                            0.4,
                        ]
                    ).reshape(-1, 1)
                )
                * self.calcWeights()
            )  # pesos personalizados resultado do produto escalar dos vetores para o cálculo dos
        else:
            self.weights = weigths * self.calcWeights()

    def calcFeaturesIdx(self, features):  # calcular os ids dos atributos selecionados
        return (
            [self.user_profile.columns.get_loc(feature) for feature in features],
            [
                self.user_profile.columns.get_loc(feature)
                for feature in features
                if self.user_profile[feature].all() == 1
            ],
        )

    def calcFeaturesWeigth(
        self,
        features,
        augmentor: float,
        reductor: float,
        lower_bound_feature: bool = False,
    ):
        """
        Calcular um fator redutor diferentes para os pesos
        das categorias que estão abaixo da escolhida
        """

        featuresIdx, selectedFeaturesIdx = self.calcFeaturesIdx(features)
        res = []

        for idx in featuresIdx:
            if idx in selectedFeaturesIdx:
                res.append(augmentor)
            else:
                if lower_bound_feature and idx < max(selectedFeaturesIdx):
                    for row in range(len(self.items_profile)):
                        # setar os atributos lower bound de user and items profile como 1 para permitir o fator redutor
                        self.items_profile.iloc[row, idx] = 1
                    self.user_profile.iloc[0, idx] = 1
                    res.append(reductor * 0.5)
                    continue
                res.append(reductor)

        return res

    def calcWeights(self):  # centralizar e vetorizar os pesos
        weightVector = [
            self.calcFeaturesWeigth(
                features=["Cachorro", "Gato", "Ave", "Coelho", "Réptil"],
                augmentor=1,
                reductor=0.333,
            ),
            self.calcFeaturesWeigth(
                features=["size_small", "size_medium", "size_large"],
                augmentor=2,
                reductor=0.5,
                lower_bound_feature=True,
            ),
            self.calcFeaturesWeigth(
                features=[
                    "isActive_True",
                    "isActive_False",
                ],
                augmentor=1,
                reductor=0.5,
            ),
            self.calcFeaturesWeigth(
                features=[
                    "expenseRange_250-499",
                    "expenseRange_500-749",
                    "expenseRange_750-999",
                    "expenseRange_1000+",
                ],
                augmentor=1,
                reductor=0.5,
                lower_bound_feature=True,
            ),
            self.calcFeaturesWeigth(
                features=["isGoodWithKids_True", "isGoodWithKids_False"],
                augmentor=1,
                reductor=0.5,
            ),
        ]
        return np.hstack(weightVector).reshape(-1, 1)  # retornar um vetor de colunas

    def similarityVector(self):
        user_vector = self.user_profile.to_numpy() * self.weights.T  # shape (1, 16)
        items_vector = (
            self.items_profile.to_numpy() * self.weights.ravel()
        )  # shape (N, 16)

        user_vector = user_vector / np.linalg.norm(user_vector)
        items_vector = items_vector / np.linalg.norm(
            items_vector, axis=1, keepdims=True
        )
        similarity = cosine_similarity(user_vector, items_vector)  # shape (1, N)
        self.cs = similarity.T  # shape (N, 1)
        return self.cs.ravel()

    def returnSimilarities(self):
        self.items_not_piped["similarities"] = self.cs
        return self.items_not_piped
