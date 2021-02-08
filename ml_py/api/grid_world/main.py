from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from base import GridWorldBase
from constants import AlgorithmTypes, grid_size
from gym_grid_world.envs import GridWorldEnv
from pg import GridWorldPG
from random import GridWorldRandom

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    AlgorithmTypes.pg: GridWorldPG(),
    AlgorithmTypes.random: GridWorldRandom()
}


@app.get('/init')
def index(algo: str):
    env = GridWorldEnv(grid_size, mode='random')
    env.reset()
    player, win, pit, wall = GridWorldBase.get_item_positions(env.state)
    model = models[algo]
    predictions = model.predict(env)
    return {"positions": {"player": player, "wall": wall, "win": win, "pit": pit}, 'grid_size': grid_size,
            'predictions': predictions}


@app.post('/step')
def step():
    pass
