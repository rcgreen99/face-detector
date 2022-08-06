from src.training.training_session_arg_parser import TrainingSessionArgParser
from src.face_detector import FaceDetector
from src.training.trainer import Trainer


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.create_model()
        self.create_trainer()
        self.trainer.run()

    def create_model(self):
        self.model = FaceDetector()

    def create_trainer(self):
        self.trainer = Trainer(
            self.model,
            self.args.dataset_path,
            self.args.batch_size,
            self.args.epochs,
            self.args.patience,
            self.args.log_dir,
            self.args.seed,
        )


if __name__ == "__main__":
    parser = TrainingSessionArgParser()
    args = parser.parse_args()
    session = TrainingSession(args)
    session.run()
