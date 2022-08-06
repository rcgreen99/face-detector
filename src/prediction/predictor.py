class Predictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, x):
        return self.model(x)
