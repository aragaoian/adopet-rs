from Database.dataExtraction import (
    userData,
    petsData,
    user_pipe,
    pets_pipe,
)
from Recommender.model import ContentBasedFiltering
from fastapi import HTTPException


def index(user_id: str):
    if not user_id.isalnum() or len(user_id) != 24:
        raise HTTPException(status_code=400, detail="User_id is malformed")

    adoptantData = userData(user_id)
    if adoptantData.empty:
        raise HTTPException(status_code=404, detail="User not found")

    user_transformed = user_pipe.fit_transform(adoptantData)  # pipeline de dados
    pets = petsData()
    pets_transformed = pets_pipe.fit_transform(pets)

    recommender = ContentBasedFiltering(  # modelo de recomendação
        user_profile=user_transformed,
        items_profile=pets_transformed,
        items_not_piped=pets,
    )
    recommender.similarityVector()
    recommended_pets = recommender.returnSimilarities()

    thresholded_recommendations = (
        recommended_pets[  # limitar a similaridades maiores que 0.85
            recommended_pets["similarities"] > 0.85
        ].sort_values("similarities", ascending=False)["id"]
    )

    response = {"recommendedPets": []}
    for recommended_pet in thresholded_recommendations:
        response["recommendedPets"].append(recommended_pet)

    return response
