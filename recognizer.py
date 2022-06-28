import utils
import constants as consts
from naive_bayes import NaiveBayes
from perceptron import Perceptron
import typer
from typing import Optional
from os import system
app = typer.Typer()

def load_prepare_data():
    
    tr_c, v_c, ts_c = utils.load_files_content()
    
    tr_im_c, tr_lb_c = tr_c
    v_im_c, v_lb_c = v_c
    ts_im_c, ts_lb_c = ts_c
    
    tr_lbls = utils.remove_nl(tr_lb_c)
    v_lbls = utils.remove_nl(v_lb_c)
    ts_lbls = utils.remove_nl(ts_lb_c)
        
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


@app.command()
def fit_bayes_model(laplace:int, out_filename: Optional[str]=None):
    system("cls")
    typer.secho(f'Building Naive Bayes Model for the data with smoothing factor of {laplace}:')
    if laplace <= 0:
        typer.secho("Error: Laplace smoothing factor for this Naive Bayes model must be positive")
        return
    labels, features, pairs = load_prepare_data()
    tr_pairs, _, _ = pairs
    evalBayes(tr_pairs, labels, features, laplace, 'va' ,out_filename)
    typer.echo()


@app.command()
def eval_bayes_model(laplace:int, target_set: str, out_filename: str = None):
    system("cls")
    typer.secho(f'Building Naive Bayes Model for the data with smoothing factor of {laplace}:')
    if laplace <= 0:
        typer.secho("Error: Laplace smoothing factor for this Naive Bayes model must be positive")
        return
    if target_set not in ['tr', 'ts', 'va']:
        typer.secho("invalid set!")
        typer.secho("use:\n-'tr' for training\n-'va' for validation\n-'ts' for test")
        return
    labels, features, pairs = load_prepare_data()
    tr_pairs, _, _ = pairs
    evalBayes(tr_pairs, labels, features, laplace, target_set ,out_filename)


def evalBayes(pairs, labels, features, laplaceSmoothingFactor:int, target_set:str, out_filename: str = None):
    model = NaiveBayes()
    
    target = _target(labels, features)
    
    model.fit(pairs, target['va']['l'], laplaceSmoothingFactor)
    
    target_features, target_lbls = target[target_set]
    
    score = model.eval(target_features, target_lbls)
    n = 'n'
    prompt = f'Accuracy for laplace factor of {laplaceSmoothingFactor} on {target[target_set][n]} set: {score}%'
    echo(prompt, out_filename)

@app.command()
def fit_perceptron_model(n_epochs:int, out_filename: Optional[str]=None):
    labels, features, _  =load_prepare_data()
    if n_epochs <= 0:
        print('Error: number of epochs must be greater than zero!')
        return
    return fitPerceptron(labels, features, n_epochs, out_filename)

def fitPerceptron(labels, features, n_epochs:int, target_set:str, out_filename: str = None):
    model = Perceptron()
    
    target = _target(labels, features)
    
    echo(f'Building Perceptron with {n_epochs} training epochs', None)
    model.fit(tr_features, tr_lbls, n_epochs)
    
    echo(f'Evaluating accuracy for validation set', None)
    score = model.eval(v_features, v_lbls)
    prompt = f'Accuracy for laplace factor of {laplaceSmoothingFactor} on {target[target_set][n]} set: {score}%'
    print(score)
    return n_epochs, score

def _target(labels, features):
    
    tr_lbls, v_lbls, ts_lbls = labels
    tr_features, v_features, ts_features = features
    
    target = {
        'tr' : {
            'n': 'training',
            'f' : tr_features,
            'l' : tr_lbls
            },
        'va' : {
            'n': 'validation',
            'f' : v_features,
            'l' : v_lbls
            },
        'ts' : {
            'n': 'testing',
            'f' : ts_features,
            'l' : ts_lbls
            }
    }
    return target

def echo(prompt, out_filename: str):
    if out_filename is not None:
        with open(out_filename, 'a') as f:
            f.write(f'{prompt}\n')
    else:
        typer.secho(prompt)

if __name__ == '__main__':
    app()