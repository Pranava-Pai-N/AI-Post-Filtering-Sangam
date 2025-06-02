from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from models.main import recommend_or_match
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SchemeItem(BaseModel):
    title: str
    description: str
    scheme_id: str

class QueryInput(BaseModel):
    query: str
    schemes: Optional[List[SchemeItem]] = []

@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}

@app.post("/recommend")
def recommend(input_data: QueryInput):
    user_schemes = [s.dict() for s in input_data.schemes] if input_data.schemes else []
    result = recommend_or_match(input_data.query, user_schemes)
    return result
