from torch.utils.data import Dataset


class FaceDetectionDataset(Dataset):
    def __init__(self, datafile, img_dir):
        self.datafile = datafile
        self.img_dir = img_dir
        self.annotations = []

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        return self.annotations[index]
