from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gym_grid_world.envs import GridWorldEnv
from model import grid_size, GridWorld

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

grid_world = GridWorld()


@app.get('/init')
async def index():
    env = GridWorldEnv(grid_size, mode='random')
    env.reset()
    player, win, pit, wall = GridWorld.get_item_positions(env.state)
    return {"positions": {"player": player, "wall": wall, "win": win, "pit": pit}, 'grid_size': grid_size}
