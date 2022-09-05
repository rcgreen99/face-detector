from torch.nn import MSELoss
import torch


class Trainer:
    def __init__(
        self, model, train_dataloader, val_dataloader, batch_size, epochs, optimizer
    ):
        self.model = model  # FaceDetector
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        print(f"Training on {self.device}...")
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validate()

    def train(self, epoch):
        running_loss = 0.0
        self.model.train()  # sets model into train mode
        for batch_index, batch in enumerate(self.train_dataloader):
            inputs = batch["inputs"].to(self.device)
            targets = batch["targets"].to(self.device)
            self.model.zero_grad()  # zero out gradients
            outputs = self.model(inputs)  # forward prop
            loss = self.criterion(outputs, targets)  # calculate loss
            loss.backward()  # calculate gradients
            self.optimizer.step()  # update model parameters (via GD)
            running_loss += loss.item()
            print(f"Epoch: {epoch} Batch: {batch_index} Loss: {running_loss:.3f}")

    def validate(self):
        self.model.eval()
