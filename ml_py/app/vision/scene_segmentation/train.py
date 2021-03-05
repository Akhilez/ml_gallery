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

    def __init__(self, data_path: str, sequence_size: int):
        self.data_path = data_path
        self.sequence_size = sequence_size
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

        # Make sequences of a batch
        """
        1. Stack all 7 layers into one.
        2. Chop off excess shots that didn't fit into n_shot // sequence_size
        3. Break these into (-1, sequence_size, *) tensors
        """
        sequences = torch.cat([data[key] for key in MovieScenesDataset.keys], 1)
        n_sequences = n_shots // self.sequence_size
        sequences = sequences[: n_sequences * self.sequence_size]
        sequences = sequences.view(
            (n_sequences, self.sequence_size, sum([2048, 512, 512, 512, 1, 1, 1]))
        )

        return sequences

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


def collate_fn_sequences(batch):
    return torch.cat(batch)


class SceneSegmenterModel(nn.Module):
    def __init__(self):
        super(SceneSegmenterModel, self).__init__()
        self.place_embed = nn.Sequential(nn.Linear(2048, 128), nn.ReLU())
        self.cast_embed = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.action_embed = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.audio_embed = nn.Sequential(nn.Linear(512, 128), nn.ReLU())

        self.embed_e = nn.Sequential(nn.Linear(512, 128), nn.ReLU())

        self.local_attention = nn.MultiheadAttention(embed_dim=512, num_heads=1)

        # Average out left and right, concatenate, then forward
        self.boundary_embedding = nn.Linear(128 * 2, 128)

        # Final attention to predict boundary
        self.boundary_predictor = nn.MultiheadAttention(embed_dim=128, num_heads=1)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, sequence_size, 2048 + 512 + 512 + 512)
        returns: tuple(
            tensor(batch_size, sequence_size, 128): Embeddings of the shots
            tensor(batch_size, sequence_size): 0-1 value for each shot. 1 if shot is the first shot of the scene.
        )
        """

        # 1. Find the embeddings
        embeddings = self.forward_embeddins(x)

        # 2. Find the segment boundaries with attention
        segments = self.forward_segmentation(embeddings)

    def forward_embeddings(self, x):
        # 1.1 Split 4 features
        place = x[:, :, :2048]
        cast = x[:, :, 2048 : 2048 + 512]
        action = x[:, :, 2048 + 512 : 2048 + 512 + 512]
        audio = x[:, :, 2048 + 512 + 512 :]

        # 1.2 Find embeddings for each feature
        place = self.place_embed(place)
        cast = self.cast_embed(cast)
        action = self.action_embed(action)
        audio = self.audio_embed(audio)

        # 1.3 Concatenate the features
        features = torch.cat((place, cast, action, audio), 2)

        # 1.4 Find the final embeddings
        e = self.embed_e(features)

        return e

    def forward_segmentation(self, x):
        # transpose so that the shape is sequence first instead of batch first
        x = x.transpose(1, 0)

        attn_output, attn_output_weights = self.local_attention(x, x, x)


def main():
    data_path = f"{BASE_DIR}/data/scenes/data/"
    dataset = MovieScenesDataset(data_path, 16)

    model = SceneSegmenterModel()

    loader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_sequences
    )

    x = next(enumerate(loader))

    """
    1. create e vectors of all shots
    2. For randomly selected negative scenes and positive scenes, find similarity loss
    """


if __name__ == "__main__":
    main()
