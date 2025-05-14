from fastapi import FastAPI, Depends
from Routers.RecommenderRouter import index
from Utils.validateToken import validateToken

app = FastAPI()

app.include_router(index, dependencies=[Depends(validateToken)])
