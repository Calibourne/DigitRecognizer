from prediction_model import PredictionModel, progressBar
class Perceptron(PredictionModel):
    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate
    def fit(self, features, labels, n_epochs, randseed):
        import random as r
        r.seed(randseed)
        
        self.weights = {}
        for i in range(10):
            self.weights[i] = [
                0
                for _ in features[0]
            ]
            # self.weights[i].append(-len(features[0])*len(features[0]))
        for i in range(n_epochs):
            
            processes = list(range(0, len(features)))
            for l in progressBar(processes, prefix=f"Progress:", suffix="Completed", length=50):
                likelyhood = {}
            # for _ ,sample in enumerate(features):
                for j in range(len(self.weights)):
                    likelyhood[j] = [
                        self.weights[j][k] * features[l][k]
                        for k in range(len(self.weights[0]))
                    ]
                    likelyhood[j] = sum(likelyhood[j])
                likelyhood = list(likelyhood.items())
                likelyhood.sort(key=lambda x: x[1], reverse=True)
            # print(f"{self.weights}")
            most_likely = likelyhood[0][0]
            true = int(labels[l])
            if most_likely != true:
                for k in range(len(self.weights[0])):
                    self.weights[most_likely][k] -= features[most_likely][k]
            # else:
                # for k in range(len(self.weights[0])):
                    self.weights[true][k] += features[true][k]

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

    def __str__(self):
        return "Perceptron Model"