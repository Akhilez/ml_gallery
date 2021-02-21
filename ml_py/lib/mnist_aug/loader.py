from torch.utils.data import Dataset
import torch

from lib.mnist_aug.mnist_augmenter import MNISTAug, DataManager


class MNISTAugDataset(Dataset):
    def __init__(
        self,
        n_out: int,
        test_mode: bool = False,
        aug: MNISTAug = None,
        noisy: bool = False,
        get_class_captions: bool = False,
        get_relationships: bool = False,
        get_positional_labels: bool = False,
        get_positional_relationships: bool = False,
        get_relationship_captions: bool = False,
    ):
        self.data_manager = DataManager()
        self.data_manager.load()

        self.aug = aug if aug is not None else MNISTAug()

        self.n_out = n_out

        x, y = (
            (self.data_manager.x_test, self.data_manager.y_test)
            if test_mode
            else (self.data_manager.x_train, self.data_manager.y_train)
        )

        x, y = self.aug.get_augmented(
            x,
            y,
            n_out=n_out,
            noisy=noisy,
            get_class_captions=get_class_captions,
            get_relationships=get_relationships,
            get_positional_labels=get_positional_labels,
            get_positional_relationships=get_positional_relationships,
            get_relationship_captions=get_relationship_captions,
        )

        self.x = torch.tensor(x, dtype=torch.float32).view((-1, 1, 112, 112))
        self.y = y

    def __len__(self):
        return self.n_out

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}
