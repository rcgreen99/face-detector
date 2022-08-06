class Trainer:
    def __init__(self, model, dataset, batch_size, epochs, patience, log_dir, seed):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.log_dir = log_dir
        self.seed = seed

    def run(self):
        pass
