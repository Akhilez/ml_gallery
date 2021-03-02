import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from settings import BASE_DIR


class MovieScenesDataset(Dataset):

    keys = [
        "place",
        "cast",
        "action",
        "audio",
        "scene_transition_boundary_ground_truth",
        "segment_ids",
    ]
    unused_keys = ["shot_end_frame", "scene_transition_boundary_prediction"]

    def __init__(self, data_path):
        self.data_path = data_path
        self.files = os.listdir(data_path)
        self.files = [file_name for file_name in self.files if file_name[-4:] == ".pkl"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(f"{self.data_path}/{self.files[index]}", "rb") as f:
            data = pickle.load(f)
        data["segment_ids"] = self.get_segment_ids(
            data["scene_transition_boundary_ground_truth"]
        )
        for key in MovieScenesDataset.unused_keys:
            del data[key]
        return data

    @staticmethod
    def get_segment_ids(segments):
        ids = [0]
        id = 0
        for segment_barrier in segments:
            if segment_barrier == 1:
                id += 1
            ids.append(id)
        return torch.tensor(ids)


def collate_fn(batch):
    collate_batch = {key: [] for key in MovieScenesDataset.keys}
    for batch_i in batch:
        for key in MovieScenesDataset.keys:
            collate_batch[key].append(batch_i[key])
    return collate_batch


def main():
    data_path = f"{BASE_DIR}/data/scenes/data/"
    dataset = MovieScenesDataset(data_path)

    loader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )

    x = next(enumerate(loader))

    print()


if __name__ == "__main__":
    main()
