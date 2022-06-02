from naive_bayes import NaiveBayes
from perceptron import Perceptron
import utils

def load_data():
    tr_im_f = open("Data/trainingimages", "r")
    tr_lb_f = open("Data/traininglabels", "r")
    
    v_im_f = open("Data/validationimages", "r")
    v_lb_f = open("Data/validationlabels", "r")
    
    ts_im_f = open("Data/testimages", "r")
    ts_lb_f = open("Data/testlabels", "r")
    
    tr_im_c, tr_lb_c = tr_im_f.readlines(), tr_lb_f.readlines()
    v_im_c, v_lb_c = v_im_f.readlines(), v_lb_f.readlines()
    ts_im_c, ts_lb_c = ts_im_f.readlines(), ts_lb_f.readlines()
    
    tr_lb_c = list(map(int,utils.remove_nl(tr_lb_c))) 
    v_lb_c = list(map(int,utils.remove_nl(v_lb_c)))
    ts_lb_c = list(map(int,utils.remove_nl(ts_lb_c)))
    
    tr_im_c = utils.join_K_rows(utils.convert_to_bits(tr_im_c))
    v_im_c = utils.join_K_rows(utils.convert_to_bits(v_im_c))
    ts_im_c = utils.join_K_rows(utils.convert_to_bits(ts_im_c))
    
    tr_pairs = utils.pair_label2features(tr_lb_c, tr_im_c)
    v_pairs = utils.pair_label2features(v_lb_c, v_im_c)
    ts_pairs = utils.pair_label2features(ts_lb_c, ts_im_c)


if __name__ == '__main__':
    load_data()