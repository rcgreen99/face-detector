from argparse import ArgumentParser


class TrainingSessionArgParser(ArgumentParser):
    def __init__(self):
        super().__init__(prog="train.sh", description="Train prediction model")
        self.add_argument(
            "--dataset_path",
            type=str,
            default="data/list_bbox_celeba.txt",
            help="path to master_dataset.csv file",
        )
        self.add_argument(
            "--img_dir",
            type=str,
            default="data/imgs",
            help="path to images directory",
        )
        self.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="number of training examples to use per batch",
        )
        self.add_argument(
            "--epochs",
            type=int,
            default=4,
            help="number of full training passes over the entire dataset",
        )
        self.add_argument(
            "--val_percent",
            type=float,
            default=0.20,
            help="percentage of dataset to use for validation",
        )
        self.add_argument(
            "--learning_rate",
            type=float,
            default=0.001,
            help="learning rate for the optimizer",
        )
        self.add_argument(
            "--patience",
            type=int,
            default=2,
            help="number of epochs without improvement after which training stops",
        )
        self.add_argument(
            "--log_dir",
            type=str,
            help="directory where training progress and model checkpoints are saved",
        )
        self.add_argument(
            "--seed",
            type=int,
            default=None,
            help="seed for random number generator",
        )
