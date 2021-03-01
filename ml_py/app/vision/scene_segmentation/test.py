from settings import BASE_DIR
import os

data_path = f"{BASE_DIR}/data/scenes/data/"

files = os.listdir(data_path)
print(files)
