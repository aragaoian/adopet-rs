from sqlalchemy import select
import pandas as pd
from Database.dbEngineSession import Session, metadata
from Recommender.defaultPipeline import TransformationPipeline

user = metadata.tables["User"]
pets = metadata.tables["Pet"]
breed = metadata.tables["Breed"]
specie = metadata.tables["Specie"]


def userData(userId: str):
    with Session() as session:
        query = select(
            user.c.id,
            user.c.petPreference,
            user.c.size,
            user.c.isActive,
            user.c.expenseRange,
            user.c.isGoodWithKids,
        ).where(user.c.id == userId)

        res = session.execute(query)
        return pd.DataFrame(res)


def petsData():
    with Session() as session:
        query = select(
            pets.c.id,
            breed.c.specieName,
            pets.c.size,
            pets.c.isActive,
            pets.c.expenseRange,
            pets.c.isGoodWithKids,
        ).where(breed.c.name == pets.c.breedName)

        res = session.execute(query)
        return pd.DataFrame(res)


def speciesList():
    with Session() as session:
        query = select(specie)
        res = list(session.execute(query))
        return list(specie[0] for specie in res)


user_pipe = TransformationPipeline(
    True,
    speciesList(),
    ["petPreference"],
    ["size", "isActive", "expenseRange", "isGoodWithKids"],
).pipeline()

pets_pipe = TransformationPipeline(
    False,
    speciesList(),
    ["specieName"],
    ["size", "isActive", "expenseRange", "isGoodWithKids"],
).pipeline()
