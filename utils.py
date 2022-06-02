import constants
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

def join_K_rows(lst: list, K = constants.ROWS_COUNT):
    K_lines = range(0,len(lst), K)
    joined = []
    for num in range(0,len(lst), K):
        if num in K_lines:
            joined.append(
                list(map(int,
                    list(''.join(lst[num:num+constants.ROWS_COUNT]))
                ))
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
