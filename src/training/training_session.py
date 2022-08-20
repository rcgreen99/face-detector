from logging.config import valid_ident
from pickletools import optimize
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

from src.training.training_session_arg_parser import TrainingSessionArgParser
from src.dataset_splitter import DatasetSplitter
from src.face_detector import FaceDetector
from src.face_detection_dataset import FaceDetectionDataset
from src.training.trainer import Trainer


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.seed_generators()
        self.create_datasets()
        self.create_dataloaders()
        self.create_model()
        self.create_optimizer()
        self.create_trainer()
        self.trainer.run()

    def seed_generators(self):
        """Seed all random number generators."""
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

    def create_datasets(self):
        dataset = FaceDetectionDataset(self.args.dataset_path, self.args.img_dir)
        self.train_dataset, self.val_dataset = DatasetSplitter(dataset).random(
            self.args.val_percent
        )

    def create_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size
        )

    def create_model(self):
        self.model = FaceDetector()

    def create_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=self.args.learning_rate)

    def create_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            optimizer=self.optimizer,
        )


if __name__ == "__main__":
    parser = TrainingSessionArgParser()
    args = parser.parse_args()
    session = TrainingSession(args)
    session.run()
