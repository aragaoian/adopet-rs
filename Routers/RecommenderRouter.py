from fastapi import APIRouter
from Controllers import RecommenderController

router = APIRouter()


@router.get("/recommendations/{user_id}")
async def index(user_id: str):
    return RecommenderController.index(user_id=user_id)
