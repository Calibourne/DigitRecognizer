from pathlib import Path
ROWS_COUNT = 28

training = "training"
validation = "validation"
testing = "testing"

digit_prob_path = "Data/digitsprob_"
tr_digits_prob = Path(f"{digit_prob_path}{training}.obj")
v_digits_prob = Path(f"{digit_prob_path}{validation}.obj")
ts_digits_prob = Path(f"{digit_prob_path}{testing}.obj")

features_path = "Data/features_"
tr_features = Path(f"{features_path}{training}.obj")
v_features = Path(f"{features_path}{validation}.obj")
ts_features = Path(f"{features_path}{testing}.obj")

pairing_path = "Data/pairing_"
tr_pairing = Path("{pairing_path}{training}.obj")
v_pairing = Path("{pairing_path}{validation}.obj")
ts_pairing = Path("{pairing_path}{testing}.obj")

BAYES = '1'
PERCEPTRON = '2'