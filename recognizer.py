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
    labels, features, pairs = load_prepare_data()
    tr_pairs, _, _ = pairs
    system("cls")
    typer.secho(f'Building Naive Bayes Model for the data with smoothing factor of {laplace}:')
    fitBayes(tr_pairs, labels, features, laplace ,out_filename)
    typer.echo()


@app.command()
def eval_bayes_model(laplace:int, target_set: str, out_filename: str = None):
    if target_set == 'tr':
        pass
    
    elif target_set == 'va':
        pass
    elif target_set == 'ts':
        pass
    else:
        typer.secho("invalid set!")
        typer.secho("use:\n-'tr' for training\n-'va' for validation\n-'ts' for test")
    pass


def fitBayes(pairs, labels, features, laplaceSmoothingFactor, out_filename: str):
    model = NaiveBayes()
    
    tr_lbls, v_lbls, _ = labels
    _, v_features, _ = features
    
    model.fit(pairs, tr_lbls, laplaceSmoothingFactor)
    score = model.eval(v_features, v_lbls)
    prompt = f'Accuracy for laplace factor of {laplaceSmoothingFactor}: {score}'
    
    if out_filename is not None:
        with open(out_filename, 'a') as f:
            f.write(f'{prompt}\n')
    else:
        typer.secho(prompt)
    
if __name__ == '__main__':
    app()