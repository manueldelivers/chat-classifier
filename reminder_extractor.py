import nltk
import re
from spellchecker import SpellChecker

FORBIDDENWORDS = ['reminders', 'reminder', 'reminds', 'remind']
REGEX = "\\b[0-9a-zA-Z']{1,}\\b"  # TODO: Update this for punctuation

spell = SpellChecker()

# TODO fix case
#  https://towardsdatascience.com/truecasing-in-natural-language-processing-12c4df086c21  # noqa

# TODO find named entities (use spacy)

# TODO: Improve grammar
#  Run `nltk.help.upenn_tagset()` to check tags or go to
#  https://www.guru99.com/pos-tagging-chunking-nltk.html
GRAMMAR = r"""
    J:
        {<JJ|JJR|JJS>*}
    N:
        {<NN|NNS|NNP|NNPS>*}
    P:
        {<PRP|PRP\$>*}
    R:
        {<RB|RBR|RBS|RP>*}
    V:
        {<VB|VBG|VBD|VBD|VBN|VBP|VBZ>*}
    NBAR:
        {<P|DT>*<N.*|J.*|>*<N.*|VBG>}
    NBAREXT:
        {<CC|IN><NBAR>}
    NP:
        {<NBAR|NBAREXT>*}
    VBAR:
        {<V><V>*<R>*<TO>*}
        {<V><V>*<R>*<TO>*}
        {<V><V>*<R|TO>*}
    ACTION:
        {<VBAR.*|NP.*|R.*>*}
    VP:
        {<ACTION><IN.*|CC.*|ACTION.*>*}
        {<ACTION>}
"""

chunker = nltk.RegexpParser(GRAMMAR)


def _pos_tag(text):
    toks = nltk.regexp_tokenize(text, REGEX)
    return nltk.tag.pos_tag(toks)


def _extractor(text):
    text = [spell.correction(w) for w in text.split()]
    if text[0] in FORBIDDENWORDS:
        text = text[1:]
    if text[-1] in FORBIDDENWORDS:
        text = text[:-1]
    text = ' '.join(text)
    toks = nltk.regexp_tokenize(text, REGEX)
    postoks = nltk.tag.pos_tag(toks)
    tree = chunker.parse(postoks)
    extractions = [' '.join(list(zip(*leaf))[0]) for leaf in _leaves(tree)]
    # TODO: drop extractions such as 'remind' 'reminder'
    extractions = [e for e in extractions if not any(w in e for w in FORBIDDENWORDS)]
    extractions = re.sub(' +', ' ', ' '.join(extractions)).strip()
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
