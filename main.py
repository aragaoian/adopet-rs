import pandas as pd
from defaultPipeline import TransformationPipeline
from model import ContentBasedFiltering

pets = [
    "dog",
    "cat",
    "rabbit",
    "bird",
    "reptile",
]

user_data = [[1, ["dog", "rabbit"], "small", "yes", "750-999", "yes"]]
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

user_pipe = TransformationPipeline(
    pets, ["animal"], ["size", "isActive", "expenseRange", "goodWithKids"]
).pipeline()

pets_pipe = TransformationPipeline(
    pets,
    ["animal", "species"],
    ["size", "isActive", "expenseRange", "goodWithKids"],
).pipeline()


user_transformed = user_pipe.fit_transform(user_df)
# pets_transformed = pets_pipe.fit_transform(pets_df)

print(f"USER: {user_transformed}")
# print(f"PETS: {pets_transformed.shape}")

# recommender = ContentBasedFiltering(
#     user_profile=user_transformed, items_profile=pets_transformed
# )
# recommender.similarityVector()
# recommended_pets = recommender.returnMatches(threshold=0.55)

# print("Recommended pet IDs:", recommended_pets["ids"])
# print("Similarity scores:", recommended_pets["similarity"])
