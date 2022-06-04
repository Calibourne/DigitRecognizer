import utils
import constants as consts
import pickle
from naive_bayes import NaiveBayes
from perceptron import Perceptron
from numpy import unique, asarray
from pathlib import Path

def load_prepare_data():
    
    tr_c, v_c, ts_c = utils.load_files_content()
    
    tr_im_c, tr_lb_c = tr_c
    v_im_c, v_lb_c = v_c
    ts_im_c, ts_lb_c = ts_c
    
    if not Path.exists(consts.tr_digits_prob) or not Path.exists(consts.v_digits_prob) or not Path.exists(consts.ts_digits_prob):
        tr_lb_c = utils.save_digit_probabilitiies(tr_lb_c, consts.training)
        v_lb_c  = utils.save_digit_probabilitiies(v_lb_c, consts.validation)
        ts_lb_c = utils.save_digit_probabilitiies(ts_lb_c, consts.testing)
    else:
        pass
        
    if not Path.exists(consts.tr_features) or not Path.exists(consts.v_features) or not Path.exists(consts.ts_features):    
        tr_im_c = utils.save_features(tr_im_c, consts.training)
        v_im_c  = utils.save_features(v_im_c, consts.validation)
        ts_im_c = utils.save_features(ts_im_c, consts.testing)
    else:
        pass

    if not Path.exists(consts.tr_pairing) or not Path.exists(consts.v_pairing) or not Path.exists(consts.ts_pairing):
        tr_pairs = utils.save_pairs(tr_lb_c, tr_im_c, consts.training)
        v_pairs  = utils.save_pairs(v_lb_c, v_im_c, consts.validation)
        ts_pairs = utils.save_pairs(ts_lb_c, ts_im_c, consts.testing)
    else:
        pass
    
    


if __name__ == '__main__':
    print(consts.digit_prob_path)
    load_prepare_data()