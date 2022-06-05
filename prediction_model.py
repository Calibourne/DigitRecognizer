from utils import progressBar
class PredictionModel:
    def predict_X(self, X):
        pass
    def evaluate(self, samples, realValues):
        TP = 0
        
        processes = list(range(0, len(samples)))
        for i in progressBar(processes, prefix=f"Progress:", suffix="Completed", length=50):
        # for idx,sample in enumerate(data):
            TP +=  1 if self.predict_X(samples[i]) == int(realValues[i]) else 0
        return (TP / len(samples))*100