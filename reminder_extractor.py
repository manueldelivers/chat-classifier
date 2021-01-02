import nltk

REGEX = "\\b[0-9a-zA-Z]{1,}\\b"  # TODO: Update this for punctuation

# TODO: Improve grammar
GRAMMAR = r"""
    NBAR:
        {<PRP$|DT><NN.*|JJ|JJR>*<NN.*>}  
        {<NN.*|JJ|JJR>*<NN.*>}  
    NP:
        {<NBAR><IN><NBAR>}
        {<NBAR>}
    ACT:
        {<VB|VBP|VBD>*<NP><VBD|VBN>}
        {<VB|VBP|VBD>*<NP><VBD|VBN>}
        {<VB|VBP|VBD>*<NP>}
    VP:
        {<ACT><IN|CC><ACT>}
        {<ACT>}
"""

chunker = nltk.RegexpParser(GRAMMAR)


def _pos_tag(text):
    toks = nltk.regexp_tokenize(text, REGEX)
    return nltk.tag.pos_tag(toks)


def _extractor(text):
    toks = nltk.regexp_tokenize(text, REGEX)
    postoks = nltk.tag.pos_tag(toks)
    tree = chunker.parse(postoks)
    extractions = [' '.join(list(zip(*leaf))[0]) for leaf in _leaves(tree)]
    # TODO: drop extractions such as 'remind' 'reminder'
    # print(text, [t for t in tree])
    return extractions


def _leaves(tree):
    for subtree in tree.subtrees(filter=lambda t: t.label() in ['VP']):
        yield subtree.leaves()


if __name__ == '__main__':
    from utils import read_csv, change_dict
    from pprint import pprint

    reminders = [row for row in read_csv('training_slots (2).csv') if
                 row['class'] == 'set_reminder']
    reminders = [change_dict(row, 'extractions', _extractor(row['utterance']))
                 for row in reminders]
    # reminders = [change_dict(row, 'POS', _pos_tag(row['utterance']))
    #              for row in reminders]
    pprint(reminders)
