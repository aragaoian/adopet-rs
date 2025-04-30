import pandas as pd

pets = [
    "dog",
    "cat",
    "bird",
    "rabbit",
    "reptile",
]

user_data = [[1, ["dog", "rabbit"], "small", "yes", "750-999", "yes"]]
user_df = pd.DataFrame(
    user_data,
    columns=["id", "animal", "size", "isActive", "expenseRange", "isGoodWithKids"],
)

pets_data = [
    [1, "dog", "Labrador Retriever", "large", "yes", "750-999", "yes"],
    [2, "dog", "Poodle", "medium", "yes", "500-749", "yes"],
    [3, "dog", "Bulldog", "medium", "no", "500-749", "yes"],
    [4, "dog", "Chihuahua", "small", "yes", "200-499", "no"],
    [5, "dog", "German Shepherd", "large", "yes", "750-999", "yes"],
    [6, "dog", "Golden Retriever", "large", "yes", "750-999", "yes"],
    [7, "cat", "Siamese", "small", "yes", "200-499", "yes"],
    [8, "cat", "Maine Coon", "large", "no", "500-749", "yes"],
    [9, "cat", "Persian", "small", "no", "200-499", "no"],
    [10, "cat", "Bengal", "medium", "yes", "500-749", "no"],
    [11, "bird", "Parakeet", "small", "yes", "200-499", "yes"],
    [12, "bird", "Canary", "small", "yes", "200-499", "yes"],
    [13, "bird", "Cockatiel", "small", "yes", "200-499", "yes"],
    [14, "bird", "Macaw", "large", "yes", "750-999", "no"],
    [15, "rabbit", "Netherland Dwarf", "small", "yes", "200-499", "yes"],
    [16, "rabbit", "Lionhead", "small", "yes", "200-499", "yes"],
    [17, "rabbit", "Flemish Giant", "large", "no", "500-749", "yes"],
    [18, "reptile", "Bearded Dragon", "medium", "no", "500-749", "no"],
    [19, "reptile", "Ball Python", "medium", "no", "500-749", "no"],
    [20, "reptile", "Leopard Gecko", "small", "no", "200-499", "no"],
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
        "isGoodWithKids",
    ],
)
