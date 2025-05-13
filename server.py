from fastapi import FastAPI, Depends
from Routers.RecommenderRouter import recommenderRouter
from Utils.validateToken import validateToken

app = FastAPI()

app.include_router(recommenderRouter, dependencies=[Depends(validateToken)])