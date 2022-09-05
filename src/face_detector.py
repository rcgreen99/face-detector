import torch.nn as nn


class FaceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)

    def __call__(self, inputs):
        pass
