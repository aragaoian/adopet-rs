from Recommender.model import ContentBasedFiltering
import numpy as np
import pandas as pd

# TODO
# adicionar métricas como MAP e NDCG and Hit Precision


class Evaluate:
    def __init__(self, user_profile_path, items_profile_path, threshold=0.50):
        self.user_profiles = pd.read_csv(user_profile_path)
        self.items_raw = pd.read_csv(items_profile_path)
        self.items_profile = self.items_raw.iloc[:, 0:16]
        self.items_relevance = self.items_raw.iloc[:, 16:]
        self.precisions = []
        self.recalls = []
        self.accuracies = []
        self.threshold = threshold
        self.columns = [
            "Cachorro",
            "Gato",
            "Ave",
            "Coelho",
            "Réptil",
            "size_small",
            "size_medium",
            "size_large",
            "isActive_True",
            "isActive_False",
            "expenseRange_250-499",
            "expenseRange_500-749",
            "expenseRange_750-999",
            "expenseRange_1000+",
            "isGoodWithKids_True",
            "isGoodWithKids_False",
        ]

    def precision(self, tp, fp):
        if tp + fp == 0:
            return 0
        elif fp == 0:
            return 1.0  # precisão perfeita, sem falso positivos
        else:
            return tp / (tp + fp)

    def recall(self, tp, fn):
        if tp + fn == 0:
            return 0
        elif fn == 0:
            return 1.0  # recall perfeito, sem falso negativos
        else:
            return tp / (tp + fn)

    def accuracy(self, tp, fp, tn, fn):
        if tp + tn + fp + fn == 0:
            return 0
        elif fp == 0 and fn == 0:
            return 1.0
        else:
            return (tp + tn) / (tp + tn + fp + fn)

    def map(self):
        raise NotImplementedError("Method not implemented!")

    def ndcg(self):
        raise NotImplementedError("Method not implemented!")

    def hitPrecision(self):
        raise NotImplementedError("Method not implemented!")

    def calculate(self):
        similarities_arr = []
        for i in range(self.user_profiles.shape[0]):
            user_profile = pd.DataFrame(
                np.array(self.user_profiles.iloc[i]).reshape(-1, 1).T,
                columns=self.columns,
            )
            c = ContentBasedFiltering(user_profile, self.items_profile, None)
            sims = c.similarityVector()
            similarities_arr.append(sims)

            tp = [
                item_idx
                for item_idx, item in enumerate(sims)
                if (
                    item > self.threshold
                    and self.items_relevance.iloc[item_idx, i] == 1
                )
            ]
            fp = [
                item_idx
                for item_idx, item in enumerate(sims)
                if (
                    item > self.threshold
                    and self.items_relevance.iloc[item_idx, i] == 0
                )
            ]
            tn = [
                item_idx
                for item_idx, item in enumerate(sims)
                if (
                    item < self.threshold
                    and self.items_relevance.iloc[item_idx, i] == 0
                )
            ]
            fn = [
                item_idx
                for item_idx, item in enumerate(sims)
                if (
                    item < self.threshold
                    and self.items_relevance.iloc[item_idx, i] == 1
                )
            ]

            self.precisions.append(self.precision(len(tp), len(fp)))
            self.recalls.append(self.recall(len(tp), len(fn)))
            self.accuracies.append(self.accuracy(len(tp), len(fp), len(tn), len(fn)))

    def printResults(self):
        print(f"Precisions: \n{self.precisions}")
        print(f"Recalls: \n{self.recalls}")
        print(f"Accuracies: \n{self.accuracies}")

        print(f"Mean precision: {sum(self.precisions) / len(self.precisions)}")
        print(f"Mean recall: {sum(self.recalls) / len(self.recalls)}")
        print(f"Mean accuracy: {sum(self.accuracies) / len(self.accuracies)}")

    def showPrecisionRecallCurve(self):
        raise NotImplementedError("Method not implemented!")


# Precisions:
# [0.4782608695652174, 0.42028985507246375, 0.7246376811594203, 0.6376811594202898, 0.4782608695652174, 0.7971014492753623, 0.855072463768116, 0.7763157894736842, 0.5072463768115942, 0.6086956521739131, 0.6052631578947368, 0.6521739130434783, 0.5072463768115942, 0.5217391304347826, 0.7818181818181819, 0.618421052631579, 0.631578947368421, 0.6956521739130435, 0.5263157894736842, 0.6571428571428571, 0.5217391304347826, 0.4605263157894737, 0.5394736842105263, 0.6376811594202898, 0.5657894736842105, 0.6285714285714286, 0.7763157894736842, 0.5921052631578947, 0.5714285714285714, 0.6376811594202898, 0.5714285714285714, 0.6666666666666666, 0.618421052631579, 0.6521739130434783, 0.6973684210526315, 0.7288135593220338, 0.6521739130434783, 0.6376811594202898, 0.6666666666666666, 0.6231884057971014, 0.6811594202898551, 0.4605263157894737, 0.7246376811594203, 0.5714285714285714, 0.5797101449275363, 0.43859649122807015, 0.5714285714285714, 0.7681159420289855, 0.4782608695652174, 0.5797101449275363]
# Recalls:
# [0.717391304347826, 0.6904761904761905, 0.5555555555555556, 0.5641025641025641, 0.6470588235294118, 0.48672566371681414, 0.47580645161290325, 0.5221238938053098, 0.660377358490566, 0.5753424657534246, 0.5974025974025974, 0.5421686746987951, 0.6363636363636364, 0.6666666666666666, 0.5180722891566265, 0.6619718309859155, 0.6666666666666666, 0.5333333333333333, 0.6896551724137931, 0.6865671641791045, 0.6666666666666666, 0.660377358490566, 0.7192982456140351, 0.5569620253164557, 0.6615384615384615, 0.676923076923077, 0.5315315315315315, 0.6521739130434783, 0.6557377049180327, 0.5789473684210527, 0.6557377049180327, 0.5542168674698795, 0.6619718309859155, 0.5421686746987951, 0.5888888888888889, 0.6825396825396826, 0.5487804878048781, 0.5641025641025641, 0.5476190476190477, 0.5657894736842105, 0.573170731707317, 0.660377358490566, 0.5555555555555556, 0.6779661016949152, 0.6666666666666666, 0.6410256410256411, 0.6557377049180327, 0.5047619047619047, 0.7021276595744681, 0.6666666666666666]
# Accuracies:
# [0.755, 0.735, 0.705, 0.705, 0.73, 0.64, 0.625, 0.645, 0.74, 0.71, 0.695, 0.69, 0.73, 0.745, 0.74, 0.735, 0.74, 0.685, 0.73, 0.775, 0.745, 0.705, 0.745, 0.7, 0.725, 0.765, 0.655, 0.725, 0.745, 0.715, 0.745, 0.7, 0.735, 0.69, 0.7, 0.82, 0.695, 0.705, 0.695, 0.705, 0.715, 0.705, 0.705, 0.755, 0.755, 0.77, 0.745, 0.66, 0.75, 0.755]

# Mean precision: 0.6156076440850904
# Mean recall: 0.6133971173898937
# Mean accuracy: 0.7197

eval = Evaluate("user_profiles.csv", "items_profile.csv")
eval.calculate()
eval.printResults()
