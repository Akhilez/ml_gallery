from decouple import config
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

print(BASE_DIR)
