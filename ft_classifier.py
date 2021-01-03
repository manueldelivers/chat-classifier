import fasttext as ft
from utils import change_dict, read_csv, write_lines
from pathlib import Path
import re

ft.FastText.eprint = lambda x: None  # suppress a deprecation warning

DATA_PATH = Path('data')
MODEL_FILEPATH = DATA_PATH / 'ft_classifier.bin'
TRAIN_FT_FILEPATH = DATA_PATH / 'train.ft'

CONTEXT_KEY = 'context'
LABEL_KEY = 'class'
TEXT_KEY = 'utterance'

CONTEXT_PREFIX = '__context__'
LABEL_PREFIX = '__label__'


class InputFileNotFoundError(Exception):
    pass


class TrainingDataNotFoundError(Exception):
    pass


class TrainedModelNotFoundError(Exception):
    pass


# `model` contains the last trained model
# if one exists then it is loaded when this file is imported
model = None
if MODEL_FILEPATH.exists():
    model = ft.load_model(str(MODEL_FILEPATH))


def _clean_context(text, context_prefix=CONTEXT_PREFIX):
    text = text.strip().lower()
    if len(text) > 0:
        text = context_prefix + text.replace(' ', '_')
    return text


def _clean_contexts(data, context_key=CONTEXT_KEY,
                    context_prefix=CONTEXT_PREFIX):
    if context_key in data[0].keys():
        data = [change_dict(row, context_key, _clean_context(
            row[context_key], context_prefix=context_prefix)) for row in data]
    return data


def _clean_labels(data, label_key=LABEL_KEY, relevant_labels=None):
    assert isinstance(relevant_labels, (str, set, list, type(None)))
    if isinstance(relevant_labels, str):
        relevant_labels = {relevant_labels}

    def filter_labels(label):
        return label if label in relevant_labels else ''

    if relevant_labels:
        relevant_labels = set(relevant_labels)
        data = [change_dict(row, label_key, filter_labels(row[label_key]))
                for row in data]

    return data


def _clean_text(text):
    # add other steps here
    text = re.sub(r"(?P<p>[^a-zA-Z0-9 ])", " \g<p> ", text)  # noqa add space for punctuation
    text = re.sub(r" +", " ", text)
    text = text.lower().strip()
    return text


def _clean_texts(data, text_key=TEXT_KEY):
    return [change_dict(row, text_key, _clean_text(row[text_key])) for row in
            data]


def _load_data(filename, text_key=TEXT_KEY, remove_empty=True):
    try:
        data = read_csv(filename)
    except FileNotFoundError:
        raise InputFileNotFoundError

    # add other conditions if needed
    # currently only ignores rows with no utterance

    if remove_empty:
        data = [row for row in data if len(row[text_key]) > 0]

    return data


def _write_ft_file(filename, text_key=TEXT_KEY, label_key=LABEL_KEY,
                   context_key=CONTEXT_KEY, remove_empty=True, clean=True,
                   relevant_labels=None, label_prefix=LABEL_PREFIX,
                   context_prefix=CONTEXT_PREFIX):
    data = _load_data(filename, text_key=text_key, remove_empty=remove_empty)

    if clean:
        data = _clean_texts(data, text_key=text_key)
        data = _clean_labels(data, label_key=label_key,
                             relevant_labels=relevant_labels)
        data = _clean_contexts(data, context_key=context_key,
                               context_prefix=context_prefix)

    def add_prefix(label):
        return f"{label_prefix}{label if len(label) > 0 else 'unknown'}"

    def make_row(label, text, context):
        row = [add_prefix(label)] + ([context] if context else []) + [text]
        return ' '.join(row)

    TRAIN_FT_FILEPATH.parent.mkdir(parents=True, exist_ok=True)

    texts = [make_row(row[label_key],
                      row[context_key] if context_key in row.keys() else '',
                      row[text_key]) for row in data]
    write_lines(texts, TRAIN_FT_FILEPATH)


def train(filename, text_key=TEXT_KEY, label_key=LABEL_KEY, remove_empty=True,
          clean=True, relevant_labels=None, prefix=LABEL_PREFIX,
          preprocess=True, multilabel=False):
    if preprocess:
        _write_ft_file(filename, text_key=text_key, label_key=label_key,
                       remove_empty=remove_empty, clean=clean,
                       relevant_labels=relevant_labels, label_prefix=prefix)

    global model

    # TODO: hyperparameters might need to be adjusted
    #  https://fasttext.cc/docs/en/python-module.html

    if not TRAIN_FT_FILEPATH.exists():
        raise TrainingDataNotFoundError

    model = ft.train_supervised(
        input=str(TRAIN_FT_FILEPATH),
        minCount=5,
        loss='ova' if multilabel else 'softmax'
    )

    MODEL_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILEPATH))


def predict(text):
    try:
        labels, probs = model.predict(_clean_text(text), k=-1)
        labels = [label.split(LABEL_PREFIX)[1] for label in labels]
        return dict(zip(labels, probs))
    except AttributeError:
        raise TrainedModelNotFoundError
