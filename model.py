from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFiltering:
    def __init__(self, user_profile, items_profile, threshold: float = 0.55):
        self.user_profile = user_profile
        self.items_profile = items_profile
        self.w = [
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
            0.4,
            0.4,
        ]
        self.threshold = threshold

    def similarityVector(self):
        self.user_profile = self.user_profile * self.w
        self.items_profile = self.items_profile * self.w
        self.cs = cosine_similarity(self.user_profile, self.items_profile)

    def returnMatches(self):
        res = {"ids": [], "similarity": []}
        for i, similarity in enumerate(self.cs.reshape((-1))):
            if similarity > self.threshold:
                res["ids"].append(i + 1)
                res["similarity"].append(similarity)
        return res
