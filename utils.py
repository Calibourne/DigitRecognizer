import constants

def load_files_content():
    tr_im = open("Data/trainingimages", "r")
    tr_lb = open("Data/traininglabels", "r")
    
    v_im = open("Data/validationimages", "r")
    v_lb = open("Data/validationlabels", "r")
    
    ts_im = open("Data/testimages", "r")
    ts_lb = open("Data/testlabels", "r")
    return (tr_im.readlines(), tr_lb.readlines()),(v_im.readlines(), v_lb.readlines()),(ts_im.readlines(), ts_lb.readlines())

def remove_nl(lst: list):
    removed = [
        r.replace("\n","") 
        for r in lst
    ]
    return removed
    
def convert_to_bits(lst: list):
    lst = remove_nl(lst)
    converted = [
        r.replace(" ", "0").replace("+", "#").replace("#","1") 
        for r in lst
    ]
    return converted

def convert_to_features(lst: list, K = constants.ROWS_COUNT):
    K_lines = range(0,len(lst), K)
    joined = []
    for num in range(0,len(lst), K):
        if num in K_lines:
            joined.append(
                list(
                    map(int, 
                        list(
                        ''.join(lst[num:num+constants.ROWS_COUNT])
                        )
                    )
                )
            )
    return joined

def pair_label2features(lbls: list, features: list):
        if len(lbls) != len(features):
            raise Exception(f"Lists do not match in length! lbls len: {len(lbls)} feats len: {len(features)}")
        paired = [
            (lbls[i], features[i])
            for i in range(len(lbls))
        ]
        return paired

def get_probabilities(lst: list):
    freq_dict = get_frequencies(lst)
    freq = list(freq_dict.values())
    probabilities = [f / len(lst) for f in freq]
    return sort_probabilities(list(freq_dict.keys()), probabilities)
    
def get_frequencies(lst: list):
    freq = dict()
    for e in lst:
        if e in freq.keys():
            freq[e] += 1
        else:
            freq[e] = 1
    return freq

def sort_probabilities(keys: list, probs: list):
    tr_paired = list(zip(keys,probs))
    tr_paired.sort(key=lambda x: x[0])
    keys, probs = tuple(zip(*tr_paired))
    return keys, probs

def save_digit_probabilitiies(lst: list, suffix: str):
    import pickle
    lst = list(map(int,remove_nl(lst))) 
    _, probs = get_probabilities(lst)
    digits = open(f"{constants.digit_prob_path}{suffix}.obj","wb")
    pickle.dump(probs, digits)
    return lst

def save_features(lst: list, suffix: str):
    import pickle
    lst = convert_to_features(convert_to_bits(lst))
    # lst = convert_to_features(lst)
    features = open(f"{constants.features_path}{suffix}.obj","wb")
    pickle.dump(lst, features)
    return lst

def save_pairs(lbls: list, feat: list, suffix: str):
    import pickle
    lst = pair_label2features(lbls, feat)
    pairs = open(f"{constants.pairing_path}{suffix}.obj","wb")
    pickle.dump(lst, pairs)
    return lst