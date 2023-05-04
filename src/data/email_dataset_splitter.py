from torch.utils.data import random_split


class EmailDatasetSplitter:
    def __init__(self, dataset):
        self.dataset = dataset

    def random_split(self, val_size=0.2) -> tuple:
        if val_size < 0 or val_size > 1:
            raise ValueError(f"val_size '{val_size}' must be between 0 and 1")
        val_size = int(len(self.dataset) * val_size)
        train_size = len(self.dataset) - val_size
        return random_split(self.dataset, [train_size, val_size])
