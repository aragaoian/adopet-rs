# import pandas as pd
# from Recommender.defaultPipeline import TransformationPipeline
# from Recommender.model import ContentBasedFiltering
# import numpy as np


# recommender = ContentBasedFiltering(
#     user_profile=user_transformed, items_profile=pets_transformed
# )
# recommender.similarityVector()
# recommended_pets = recommender.returnMatches()

# keys = list(recommended_pets.keys())
# values = list(recommended_pets.values())
# values_f = [float(item) for arr in values for item in arr]
# indexes = np.argsort(values_f)[::-1]
# s = {keys[i]: values[i] for i in indexes}
# print(s)
