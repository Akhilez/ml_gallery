from typing import Optional
import typer
import os

app = typer.Typer()


def all_projects():
    d = os.getcwd()
    dirs = list(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))
    return dirs


@app.command()
def local(name: str = typer.Option(..., autocompletion=all_projects)):
    module = os.path.dirname(__file__).replace('/', '.')
    module = f'{module}.{name}'
    os.system(f'uvicorn {module}.main:app --reload')


if __name__ == "__main__":
    typer.run(local)
