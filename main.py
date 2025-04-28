import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from pipeline import TransformationPipeline

pets = [
    "dog",
    "cat",
    "rabbit",
    "bird",
    "reptile",
]

user_data = [[1, ["dog", "rabbit"], "small", "Yes", "750-999", "Yes"]]
user_df = pd.DataFrame(
    user_data,
    columns=["id", "animal", "size", "isActive", "expenseRange", "goodWithKids"],
)

pets_data = [
    [1, "Dog", "Labrador Retriever", "large", "Yes", "750-999", "Yes"],
    [2, "Dog", "Poodle", "medium", "Yes", "500-749", "Yes"],
    [3, "Dog", "Bulldog", "medium", "No", "500-749", "Yes"],
    [4, "Dog", "Chihuahua", "small", "Yes", "200-499", "No"],
    [5, "Dog", "German Shepherd", "large", "Yes", "750-999", "Yes"],
    [6, "Dog", "Golden Retriever", "large", "Yes", "750-999", "Yes"],
    [7, "Cat", "Siamese", "small", "Yes", "200-499", "Yes"],
    [8, "Cat", "Maine Coon", "large", "No", "500-749", "Yes"],
    [9, "Cat", "Persian", "small", "No", "200-499", "No"],
    [10, "Cat", "Bengal", "medium", "Yes", "500-749", "No"],
    [11, "Bird", "Parakeet", "small", "Yes", "200-499", "Yes"],
    [12, "Bird", "Canary", "small", "Yes", "200-499", "Yes"],
    [13, "Bird", "Cockatiel", "small", "Yes", "200-499", "Yes"],
    [14, "Bird", "Macaw", "large", "Yes", "750-999", "No"],
    [15, "Rabbit", "Netherland Dwarf", "small", "Yes", "200-499", "Yes"],
    [16, "Rabbit", "Lionhead", "small", "Yes", "200-499", "Yes"],
    [17, "Rabbit", "Flemish Giant", "large", "No", "500-749", "Yes"],
    [18, "Reptile", "Bearded Dragon", "medium", "No", "500-749", "No"],
    [19, "Reptile", "Ball Python", "medium", "No", "500-749", "No"],
    [20, "Reptile", "Leopard Gecko", "small", "No", "200-499", "No"],
]
pets_df = pd.DataFrame(
    pets_data,
    columns=[
        "id",
        "animal",
        "species",
        "size",
        "isActive",
        "expenseRange",
        "goodWithKids",
    ],
)

user_df = TransformationPipeline(
    pets, user_df, ["animal"], ["size", "isActive", "expenseRange", "goodWithKids"]
).pipeline()

print(user_df)


# weights = [
#     0.6,
#     0.6,
#     0.6,
#     0.6,
#     0.6,
#     1.0,
#     1.0,
#     1.0,
#     0.2,
#     0.2,
#     0.8,
#     0.8,
#     0.8,
#     0.4,
#     0.4,
# ]
# user_df = user_df * weights
# pets_df = pets_df * weights
# cs = cosine_similarity(user_df, pets_df)

# res = {"ids": [], "similarity": []}
# for i, similarity in enumerate(cs.reshape((-1))):
#     if similarity > 0.55:
#         res["ids"].append(i + 1)
#         res["similarity"].append(similarity)

# print(res)
