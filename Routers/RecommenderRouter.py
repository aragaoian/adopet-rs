from fastapi import APIRouter, FastAPI
from Controllers import RecommenderController
from fastapi import HTTPException

router = APIRouter()

@router.get("/recommendations/{user_id}")
async def index(user_id: str):
    return RecommenderController.index(user_id=user_id)