from decimal import Decimal

from prediction_model import PredictionModel

from utils import get_probabilities
class NaiveBayes(PredictionModel):
    X = 2 # binary labels
    def fit(self,data: dict, labels: list ,laplaceSmoothingFactor: int):
        self.catProbs = get_probabilities(labels)
        self.dataProb = {}
        
        for i in range(len(set(labels))):
            sums = [sum(x) for x in zip(*data[i])]
            self.dataProb[i] = {}
            for fi,s in enumerate(sums):
                self.dataProb[i][fi] = {}
                self.dataProb[i][fi][0] = Decimal(len(data[i]) - s + laplaceSmoothingFactor) / Decimal(len(data[i]) + NaiveBayes.X*laplaceSmoothingFactor)
                self.dataProb[i][fi][1] = Decimal(s + laplaceSmoothingFactor) / Decimal(len(data[i]) + NaiveBayes.X*laplaceSmoothingFactor)


    def predict_X(self, X: list):
        probabilities = {}
        for i in range(10):
            probabilities[i] = Decimal(1)
        for j in range(len(self.catProbs)):
            for i in range(len(X)):
                prob_i = self.dataProb[j][i][X[i]]
                probabilities[j] *= Decimal(prob_i)
            probabilities[j] *= Decimal(self.catProbs[j])
        probabilities_sorted = [
            (i,j)
            for i, j in probabilities.items()
        ]
        probabilities_sorted.sort(key=lambda x: x[1], reverse=True)
        return probabilities_sorted[0][0]
    
    def eval(self, samples, realValues):
        return super().evaluate(samples, realValues)
    
    def __str__(self):
        return "Naive Bayes Model"