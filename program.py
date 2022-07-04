from os import system
from recognizer import load_prepare_data, buildBayesModel, buildPerceptron ,evalModel
from constants import SEP
from time import perf_counter

def NaiveBayes():
    start = perf_counter()
    
    system("cls")
    data = load_prepare_data()
    points = []
    
    for i in range(1,6):
        print(f'Building NB model with laplace smoothing of {i}:')
        model = buildBayesModel(data, i)
        print('Evaluating the model...')
        points.append((i, evalModel(data, model, 'va')))
        print(f'Accuracy for NB model with smoothing of {i}: {points[i-1][1]}%', end="\n\n")
        print(SEP)
    
    
    points.sort(key=lambda x: (x[1], -x[0]), reverse=True)
    best_score = points[0][0]
    model = buildBayesModel(data, best_score)
    for s in ['ts', 'tr']:
        print(f"Evaluating accuracy for NB model with smoothing of {best_score} on {data[s]['n']} set:")
        score = evalModel(data, model, s)
        print(f'Accuracy of the NB model on {data[s]["n"]} set is: {score}%', end="\n\n")
        print(SEP)
    delta = round(perf_counter() - start,2)
    print(f'Total runtime: {delta} sec')

def Perceptron():
    start = perf_counter()
    
    system("cls")
    data = load_prepare_data()
    
    print(f'Building Perceptron model with 3 epochs:')
    model = buildPerceptron(data, 3)
    for s in ['ts', 'tr']:
        print(f"Evaluating accuracy for Perceptron model with 3 epochs on {data[s]['n']} set:")
        score = evalModel(data, model, s)
        print(f'Accuracy of the Perceptron model on {data[s]["n"]} set is: {score}%', end="\n\n")
    print(SEP)
    
    for i in range(1, 6):
        print(f'Building Perceptron model with {i} epochs:')
        model = buildPerceptron(data, i)
        print()
        for s in ['ts', 'tr']:
            print(f"Evaluating accuracy for Perceptron model with {i} epochs on {data[s]['n']} set:")
            score = evalModel(data, model, s)
            print(f'Accuracy of the Perceptron model on {data[s]["n"]} set is: {score}%')
        print(SEP)
    delta = round(perf_counter() - start,2)
    print(f'Total runtime: {delta} sec')
    
if __name__ == '__main__':
    s = ""
    while s != 'q':
        s = input('Enter a model to evaluate (n for naive bayes, p for perceptron, q to exit): ')
        if s not in ['p', 'n', 'q']:
            print('Error, command not recognized. Try again')
        else:
            if s == 'n':
                NaiveBayes()
            if s == 'p':
                Perceptron()