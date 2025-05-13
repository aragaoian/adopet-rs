from Database.dataExtraction import (
    userData,
    petsData,
    user_pipe,
    pets_pipe,
)
from Recommender.model import ContentBasedFiltering
from fastapi import HTTPException
import numpy as np


def index(userId: str):
    adoptantData = userData(userId)
    if adoptantData.empty:
        raise HTTPException(status_code=404, detail="User not found")

    user_transformed = user_pipe.fit_transform(adoptantData)
    pets = petsData()
    print(pets)
    pets_transformed = pets_pipe.fit_transform(pets)

    recommender = ContentBasedFiltering(
        user_profile=user_transformed, items_profile=pets_transformed
    )
    recommender.similarityVector()
    recommended_pets = recommender.returnMatches()

    keys = list(recommended_pets.keys())
    values = list(recommended_pets.values())
    values_f = [float(item) for arr in values for item in arr]
    indexes = np.argsort(values_f)[::-1]
    sortedSimilarities = {keys[i]: values[i] for i in indexes}

    print(sortedSimilarities)

    # response = {"recommendedPets": []}
    # for petObj in sortedSimilarities:
    #     response["recommendedPets"].append(pets.iloc[petObj]["id"])

    # return response
