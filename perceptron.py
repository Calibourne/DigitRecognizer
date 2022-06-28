from prediction_model import PredictionModel, progressBar
from utils import vec_mul as w_in
class Perceptron(PredictionModel):
    # def __init__(self, learning_rate=0.5):
    #     self.learning_rate = learning_rate
    def fit(self, features, labels, n_epochs):
        self.weights = {}
        for i in range(len(set(labels))):
            self.weights[i] = [
                0
                for _ in features[0]
            ]
        # print(self.weights.keys())
        for i in range(n_epochs):
            
            processes = list(range(0, len(features)))
            print(f'running epoch {i+1}')
            for l in progressBar(processes, prefix=f"Progress:", suffix="Completed", length=50):
                likelyhood = {}
                for j in range(len(self.weights)):
                    likelyhood[j] = w_in(features[l], self.weights[j])
                likelyhood = list(likelyhood.items())
                likelyhood.sort(key=lambda x: x[1], reverse=True)
                most_likely = likelyhood[0][0]
                true = int(labels[l])
                if most_likely != true:
                    for k in range(len(self.weights[0])):
                        self.weights[most_likely][k] -= features[l][k]
                        self.weights[true][k] += features[l][k]

    def eval(self,samples, realValues):
        return super().evaluate(samples, realValues)
    
    def predict_X(self, X):
        likelyhood = {}
        for j in range(len(self.weights)):
            likelyhood[j] = [
                self.weights[j][k] * X[k]
                for k in range(len(self.weights[0]))
            ]
            likelyhood[j] = sum(likelyhood[j])
        likelyhood = list(likelyhood.items())
        likelyhood.sort(key=lambda x: x[1], reverse=True)
        return likelyhood[0][0]
    def __str__(self):
        return "Perceptron Model"