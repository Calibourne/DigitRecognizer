from decimal import Decimal

from prediction_model import PredictionModel
class NaiveBayes(PredictionModel):
    
    # ,catProb: list
    def fit(self,data: dict, laplaceSmoothingFactor: int):
        # self.catProb = catProb
        self.catProb = [
            0.1
            for i in range(10)
        ]
        self.dataProb = {}
        for i in range(10):
            self.dataProb[i] = [
            Decimal(sum(x) + laplaceSmoothingFactor) / Decimal(len(data[i]) + laplaceSmoothingFactor)
            # (sum(x)) / (len(data[i]))
            for x in zip(*data[i])
        ] # probability for '1' feature
        # print(self.dataProb)
        
    def predict_X(self, X: list):
        from math import log
        probabilities = {}
        for i in range(10):
            probabilities[i] = Decimal(1)
        for i in range(len(X)):
            for j in range(len(self.catProb)):
                prob_i = self.dataProb[j][i] if X[i] == 1 else 1-self.dataProb[j][i]
                # print(prob_i)
                # print(self.catProb[j])
                probabilities[j] *= (Decimal(log(self.catProb[j]))+Decimal(log(prob_i)))
                # probabilities[j] *= (Decimal(self.catProb[j])*Decimal(prob_i))
                # print(probabilities[j])
        probabilities_sorted = [
            (i,j)
            for i, j in probabilities.items()
        ]
        probabilities_sorted.sort(key=lambda x: x[1], reverse=False)
        # print(probabilities_sorted)
        return probabilities_sorted[0][0]
    
    def eval(self, samples, realValues):
        return super().evaluate(samples, realValues)
    
    def __str__(self):
        return "Naive Bayes Model"