def count_lines(fname):
    with open(fname,encoding="utf8") as f:
        return sum(1 for line in f)


def detokenize(tokens):
    ret = ''
    for g, a in zip(tokens['gloss'], tokens['after']):
        ret += g + a
    return ret.strip()
