from torch.nn import MSELoss


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

    def run(self):
        print("Training...")
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validate()

    def train(self, epoch):
        criterion = MSELoss()
        running_loss = 0.0
        self.model.train()  # sets model into train mode
        for batch_index, batch in enumerate(self.train_dataloader):
            image = batch["image"]
            target = batch["target"]
            self.model.zero_grad()  # zero out gradients
            outputs = self.model(image)  # forward prop
            loss = criterion(outputs, target)  # calculate loss
            loss.backward()  # calculate gradients
            self.optimizer.step()  # update model parameters (via GD)
            running_loss += loss.item()
            print(f"Epoch: {epoch} Batch: {batch_index} Loss: {running_loss:.3f}")

    def validate(self):
        self.model.eval()
