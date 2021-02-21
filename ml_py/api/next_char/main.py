from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model import NextChar

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

next_char = NextChar()


@app.get("/")
async def index(text: str):
    return {"pred": next_char.predict(text)}
