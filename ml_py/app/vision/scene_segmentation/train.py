import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from settings import BASE_DIR


class MovieScenesDataset(Dataset):

    keys = [
        "place",
        "cast",
        "action",
        "audio",
        "scene_ids",
        "movie_ids",
        "scene_transition_boundary_ground_truth",
    ]
    unused_keys = ["shot_end_frame", "scene_transition_boundary_prediction"]

    def __init__(self, data_path: str, chunk_size: int):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.files = os.listdir(data_path)
        self.files = [file_name for file_name in self.files if file_name[-4:] == ".pkl"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(f"{self.data_path}/{self.files[index]}", "rb") as f:
            data = pickle.load(f)
        labels_key = MovieScenesDataset.keys[-1]
        n_shots = len(data["place"])

        # Indices for each scene
        data["scene_ids"] = self.get_scene_ids(data[labels_key])

        # Prepending labels with 0 so that there's a 1 for every 1st frame of new scene.
        data[labels_key] = torch.cat((torch.tensor([0.0]), data[labels_key]))

        # Movie id for identification
        data["movie_ids"] = torch.tensor([index] * n_shots)

        # Remove the data that we don't need
        for key in MovieScenesDataset.unused_keys:
            del data[key]

        # Make 1D tensor to 2D
        for key in MovieScenesDataset.keys[-3:]:
            data[key] = data[key].view((-1, 1))

        # Make chunks of a batch
        """
        1. Stack all 7 layers into one.
        2. Chop off excess shots that didn't fit into n_shot // chunk_size
        3. Break these into (-1, chunk_size, *) tensors
        """
        chunks = torch.cat([data[key] for key in MovieScenesDataset.keys], 1)
        n_chunks = n_shots // self.chunk_size
        chunks = chunks[: n_chunks * self.chunk_size]
        chunks = chunks.view(
            (n_chunks, self.chunk_size, sum([2048, 512, 512, 512, 1, 1, 1]))
        )

        return chunks

    @staticmethod
    def get_scene_ids(segments):
        ids = [0]
        id = 0
        for segment_barrier in segments:
            if segment_barrier == 1:
                id += 1
            ids.append(id)
        return torch.tensor(ids)


def collate_fn_dict(batch):
    collate_batch = {key: [] for key in MovieScenesDataset.keys}
    for batch_i in batch:
        for key in MovieScenesDataset.keys:
            collate_batch[key].append(batch_i[key])
    return collate_batch


def collate_fn_chunks(batch):
    return torch.cat(batch)


class SceneSegmenterModel(nn.Module):
    def __init__(self):
        super(SceneSegmenterModel, self).__init__()

    def forward(self, x):
        """
        x: tensor of shape (batch_size, chunk_size, 2048 + 512 + 512 + 512)
        returns: tuple(
            tensor(batch_size, chunk_size, 128): Embeddings of the shots
            tensor(batch_size, chunk_size): 0-1 value for each shot. 1 if shot is the first shot of the scene.
        )
        """

        # 1. Find the embeddings

        # 1.1 Split 4 features
        # 1.2 Find embeddings for each feature
        # 1.3 Concatenate the features
        # 1.4 Find the final embeddings

        # 2. Find the segment boundaries with attention


def main():
    data_path = f"{BASE_DIR}/data/scenes/data/"
    dataset = MovieScenesDataset(data_path, 16)

    model = SceneSegmenterModel()

    loader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_chunks
    )

    x = next(enumerate(loader))

    """
    1. create e vectors of all shots
    2. For randomly selected negative scenes and positive scenes, find similarity loss
    """


if __name__ == "__main__":
    main()
