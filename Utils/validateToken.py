import os
from dotenv import load_dotenv
from fastapi import Security
from fastapi.security import APIKeyHeader
from fastapi.exceptions import HTTPException

load_dotenv()

API_HEADER = APIKeyHeader(name="Authorization", auto_error=False)
API_KEY = os.getenv("API_KEY")


def validateToken(token: str = Security(API_HEADER)):
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or Invalid token")
    return token
