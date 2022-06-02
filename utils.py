import constants
def convert_to_bits(lst: list):
    converted = [
        r.replace("\n","").replace(" ", "0").replace("+", "#").replace("#","1") 
        for r in lst
    ]
    return converted

def join_K_rows(lst: list, K = constants.ROWS_COUNT):
    K_lines = range(0,len(lst), K)
    joined = []
    for line, num in enumerate(K_lines):
        if line in K_lines:
            joined.append(
                list(map(int,
                    list(''.join(lst[num:num+constants.ROWS_COUNT]))
                ))
            )
    return joined