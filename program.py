import utils
import constants as consts
from naive_bayes import NaiveBayes
from perceptron import Perceptron
from numpy import unique, asarray
from pathlib import Path

def load_prepare_data():
    
    tr_c, v_c, ts_c = utils.load_files_content()
    
    tr_im_c, tr_lb_c = tr_c
    v_im_c, v_lb_c = v_c
    ts_im_c, ts_lb_c = ts_c
    
    tr_lbls = utils.remove_nl(tr_lb_c)
    v_lbls = utils.remove_nl(v_lb_c)
    ts_lbls = utils.remove_nl(ts_lb_c)
    
    # tr_digitsprob = utils.get_probabilities(list(map(int,tr_lb_c)))
    # v_digitsprob  = utils.get_probabilities(list(map(int,v_lb_c)))
    # ts_digitsprob = utils.get_probabilities(list(map(int,ts_lb_c)))
        
    tr_features = utils.convert_to_features(utils.convert_to_bits(tr_im_c))
    v_features  = utils.convert_to_features(utils.convert_to_bits(v_im_c))
    ts_features = utils.convert_to_features(utils.convert_to_bits(ts_im_c))
    
    tr_pairs = utils.pair_label2features(tr_lbls, tr_features)
    v_pairs  = utils.pair_label2features(v_lbls, v_features)
    ts_pairs = utils.pair_label2features(ts_lbls, ts_features)
    
    labels = (tr_lbls, v_lbls, ts_lbls)
    # digitsprobs = (tr_digitsprob, v_digitsprob, ts_digitsprob)
    features = (tr_features, v_features, ts_features)
    pairings = (tr_pairs, v_pairs, ts_pairs)
    
    return labels, features, pairings


def main():
    from os import system
    system("cls")
    print("Loading data...")
    labels, features, pairs = load_prepare_data()
    
    
    print("Loading complete!")
    
    print("Please choose a prediction model: (or q to quit)")
    print("1) Naive Bayes")
    print("2) Perceptron")
    print("q) Quit")
    
    selection = input("Your selection: ")
    while selection != 'q':
        while selection not in (consts.BAYES, consts.PERCEPTRON):
            selection = input("Try again: ")
        system("cls")
        model = NaiveBayes() if selection == consts.BAYES else Perceptron()
        print(f"Selected {model}: ")
    
        tr_pairs, _, _ = pairs
        if isinstance(model, NaiveBayes):
            BayesEval(model, tr_pairs, labels, features)
        if isinstance(model, Perceptron):
            PercepEval(model, features, labels)
        
        print("Evaluation Complete!")
        print("You may choose another model, or quit:")
        selection = input("Your selection: ")


def BayesEval(model: NaiveBayes, pairs, labels, features):
    tr_lbls, v_lbls, ts_lbls = labels
    tr_features, v_features, ts_features = features
    bestSmoothing, bestScore = 1, 0
    
    for i in range(1,6):
        model.fit(pairs, i)
        print(f"Evaluating accuracy for laplace factor of {i} on vaidation set:")
        score = model.eval(v_features, v_lbls)
        print(f"Accuracy for laplace factor of {i} : {score}", end="\n\n")
        if score > bestScore:
            bestScore = score
            bestSmoothing = i
    
    model.fit(pairs, bestSmoothing)
    
    print(f"Evaluating accuracy for testing set:")
    score = model.eval(ts_features, ts_lbls)
    print(f"Accuracy for testing set: {score}")
    
    print(f"Evaluating accuracy for training set:")
    score = model.eval(tr_features, tr_lbls)
    print(f"Accuracy for training set: {score}")

def PercepEval(model: Perceptron, features, labels):
    tr_features, v_features, ts_features = features
    tr_lbls, v_lbls, ts_lbls = labels
    
    print("Constracting perceptron with 3 epochs evaluation:")
    model.fit(tr_features, tr_lbls, 3, 10)
    print("Evaluating on validation set:")
    score = model.eval(v_features, v_lbls)
    print(f"Accuracy for validation set: {score}")

if __name__ == '__main__':
    main()
    
    # main()
    
    
    # # tr_c, v_c, ts_c = utils.load_files_content()
    
    # # tr_im_c, tr_lb_c = tr_c
    
    # # tr_feats = utils.save_features(tr_im_c, consts.training)
    # # print(len(tr_feats))
    
    # tr_, v_, ts_, = features
