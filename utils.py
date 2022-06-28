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
        paired_dict= {}
        for i in range(10):
            paired_dict[i] = []
        for lbl, feat in paired:
            paired_dict[int(lbl)].append(feat)
        return paired_dict

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
    return probs

# def save_digit_probabilitiies(lst: list, suffix: str):
#     import pickle
#     lst = list(map(int,remove_nl(lst))) 
#     probs = get_probabilities(lst)
#     digits = open(f"{constants.digit_prob_path}{suffix}.obj","wb")
#     pickle.dump(probs, digits)
#     return probs

# def save_features(lst: list, suffix: str):
#     import pickle
#     lst = convert_to_features(convert_to_bits(lst))
#     # lst = convert_to_features(lst)
#     features = open(f"{constants.features_path}{suffix}.obj","wb")
#     pickle.dump(lst, features)
#     return lst

# def save_pairs(lbls: list, feat: list, suffix: str):
#     import pickle
#     lst = pair_label2features(lbls, feat)
#     pairs = open(f"{constants.pairing_path}{suffix}.obj","wb")
#     pickle.dump(lst, pairs)
#     return lst

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        #percent = f"0.{100*(iteration / float(total))}"
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()
    
def vec_mul(vecA: list, vecB: list):
    if len(vecA) != len(vecB):
        raise ArithmeticError("vecA and vecB must be equal in length")
    mul = [vecA[i]*vecB[i] for i in range(len(vecA))]
    return sum(mul)