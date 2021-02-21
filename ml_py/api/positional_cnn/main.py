from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import PositionalCNN

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

positional_cnn = PositionalCNN()


class ImageData(BaseModel):
    image: List[List[List[List[float]]]]


@app.post("/")
async def index(data: ImageData):
    cls, pos = positional_cnn.predict(data.image)
    return {"class": cls, "position": pos}
