from sklearn.metrics.pairwise import cosine_similarity


# self.w = [
#             0.6,
#             0.2,
#             0.2,
#             0.6,
#             0.2,
#             2.0,
#             0.1,
#             0.1,
#             0.2,
#             0.1,
#             0.6,
#             0.7,
#             0.8,
#             0.1,
#             0.4,
#             0.2,
#         ]



class ContentBasedFiltering:
    def __init__(self, user_profile, items_profile, threshold: float = 0.85):
        self.user_profile = user_profile
        self.items_profile = items_profile
        self.threshold = threshold
        self.cs = 0
        self.weights = [
            0.6 * self.calcWeights(augmentor=1, reductor=0.333, lower_idx=0, feature_idx="dog", lower_bound_feature=True),
            0.6 * self.calcWeights(augmentor=1, reductor=0.333, lower_idx=0, feature_idx="cat", lower_bound_feature=True),
            0.6 * self.calcWeights(augmentor=1, reductor=0.333, lower_idx=0, feature_idx="bird", lower_bound_feature=True),
            0.6 * self.calcWeights(augmentor=1, reductor=0.333, lower_idx=0, feature_idx="rabbit", lower_bound_feature=True),
            0.6 * self.calcWeights(augmentor=1, reductor=0.333, lower_idx=0, feature_idx="reptile", lower_bound_feature=True),
            1.0 * self.calcWeights(augmentor=2, reductor=0.5, lower_idx=0, feature_idx="size_small", lower_bound_feature=True),
            1.0 * self.calcWeights(augmentor=2, reductor=0.5, lower_idx=0, feature_idx="size_medium", lower_bound_feature=True),
            1.0 * self.calcWeights(augmentor=2, reductor=0.5, lower_idx=0, feature_idx="size_large", lower_bound_feature=True),
            0.2 * self.calcWeights(augmentor=2, reductor=0.5, lower_idx=0, feature_idx="isActive_true", lower_bound_feature=True),
            0.2 * self.calcWeights(augmentor=2, reductor=0.5, lower_idx=0, feature_idx="isActive_false", lower_bound_feature=True),
            0.8 * self.calcWeights(augmentor=1, reductor=0.5, lower_idx=0, feature_idx="expenseRange_250-499", lower_bound_feature=True),
            0.8 * self.calcWeights(augmentor=1, reductor=0.5, lower_idx=0, feature_idx="expenseRange_450-749", lower_bound_feature=True),
            0.8 * self.calcWeights(augmentor=1, reductor=0.5, lower_idx=0, feature_idx="expenseRange_750-999", lower_bound_feature=True),
            0.8 * self.calcWeights(augmentor=1, reductor=0.5, lower_idx=0, feature_idx="expenseRange_1000+", lower_bound_feature=True),
            0.4 * self.calcWeights(augmentor=1, reductor=0.5, lower_idx=0, feature_idx="isGoodWithKids_true", lower_bound_feature=True),
            0.4 * self.calcWeights(augmentor=1, reductor=0.5, lower_idx=0, feature_idx="isGoodWithKids_false", lower_bound_feature=True),
        ]

    def calcWeights(self, 
                    augmentor: float, 
                    reductor: float, 
                    lower_idx: int, 
                    feature_idx: int, 
                    lower_bound_feature: bool = False):
        '''
            Calcular um fator redutor diferentes para os pesos
            das categorias que estão abaixo da escolhida
            - Tirar a redundancia
            - Enviar as colunas
        '''
        if lower_bound_feature:
            for idx in range(lower_idx, feature_idx+1):
                if idx == feature_idx:
                    return augmentor
                else:
                    return reductor * 0.5
            else:
                return reductor
        return reductor if self.user_profile[feature_idx] != 1 else augmentor
    

    def similarityVector(self):
        user_vector = self.user_profile.to_numpy() * self.w
        items_vector = self.items_profile.to_numpy() * self.w
        self.cs = cosine_similarity(user_vector, items_vector)

    def returnMatches(self):
        res = {}
        for i, similarity in enumerate(self.cs.reshape((-1, 1))):
            if similarity > self.threshold:
                res[(i + 1)] = similarity
        return res
