from fastapi import FastAPI

from next_char.model import NextChar

app = FastAPI()
next_char = NextChar()


@app.get('/')
async def index(text: str):
    return {"pred": next_char.predict(text)}
