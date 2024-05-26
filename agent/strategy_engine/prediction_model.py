from .machine_learning_model import MachineLearningModel

class PredictionModel:
    def __init__(self):
        self.ml_model = MachineLearningModel()

    def train(self, X, y):
        self.ml_model.train(X, y)

    def predict(self, X):
        return self.ml_model.predict(X)