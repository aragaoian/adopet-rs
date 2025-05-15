from fastapi import FastAPI, Depends
from Routers import RecommenderRouter
from Utils.validateToken import validateToken

app = FastAPI()

app.include_router(RecommenderRouter.router, dependencies=[Depends(validateToken)])
