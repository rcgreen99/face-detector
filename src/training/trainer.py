class Trainer:
    def __init__(
        self, model, train_dataloader, val_dataloader, batch_size, epochs, optimizer
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer

    def run(self):
        print("Training...")
