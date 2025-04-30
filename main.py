import pandas as pd
from defaultPipeline import TransformationPipeline
from model import ContentBasedFiltering
from Data.data import user_df, pets_df, pets
import numpy as np

"""
Consider user-satisfaction as metric 
"""

user_pipe = TransformationPipeline(
    pets, ["animal"], ["size", "isActive", "expenseRange", "isGoodWithKids"]
).pipeline()

pets_pipe = TransformationPipeline(
    pets,
    ["animal", "species"],
    ["size", "isActive", "expenseRange", "isGoodWithKids"],
).pipeline()


user_transformed = user_pipe.fit_transform(user_df)
pets_transformed = pets_pipe.fit_transform(pets_df)

recommender = ContentBasedFiltering(
    user_profile=user_transformed, items_profile=pets_transformed
)
recommender.similarityVector()
recommended_pets = recommender.returnMatches()

keys = list(recommended_pets.keys())
values = list(recommended_pets.values())
values_f = [float(item) for arr in values for item in arr]
indexes = np.argsort(values_f)[::-1]
s = {keys[i]: values[i] for i in indexes}
print(s)
