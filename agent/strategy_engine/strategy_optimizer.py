from .prediction_model import PredictionModel

class StrategyOptimizer:
    def __init__(self):
        self.prediction_model = PredictionModel()

    def optimize(self, X, y):
        self.prediction_model.train(X, y)
        predictions = self.prediction_model.predict(X)
        return predictions