from torch.utils.data import random_split


class DatasetSplitter:
    def __init__(self, dataset):
        self.dataset = dataset

    def random(self, val_percent):
        if val_percent < 0.0 or val_percent > 1.0:
            raise ValueError(f"val_percent '{val_percent}' must be between 0.0 and 1.0")

        val_size = int(len(self.dataset) * val_percent)
        train_size = len(self.dataset) - val_size

        return random_split(self.dataset, [train_size, val_size])
