from __future__ import annotations
import os
from torch.utils.data import Dataset
from src.annotation import Annotation


class FaceDetectionDataset(Dataset):
    def __init__(self, datafile, img_dir):
        self.datafile = datafile
        self.img_dir = img_dir
        self.annotations = []

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        return self.annotations[index]

    def load_data(self):
        for index, line in enumerate(open(self.datafile)):
            if index < 1:
                continue
            parts = line.split()
            image_path = os.path.join(self.img_dir, parts[0])
            annotation = Annotation(
                image_path, int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            )
            self.annotations.append(annotation)
