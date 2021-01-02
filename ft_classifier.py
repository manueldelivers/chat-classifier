import fasttext as ft
from utils import change_dict, read_csv, write_lines
import os
from pathlib import Path

ft.FastText.eprint = lambda x: None  # suppress a deprecation warning

DATA_PATH = Path('data')
MODEL_FILEPATH = DATA_PATH / 'ft_classifier.bin'
TRAIN_FT_FILEPATH = DATA_PATH / 'train.ft'

LABEL_KEY = 'class'
TEXT_KEY = 'utterance'

PREFIX = '__label__'


class TrainedModelNotFound(Exception):
    pass


# `model` contains the last trained model
# if one exists then it is loaded when this file is imported
model = None
if os.path.exists(MODEL_FILEPATH):
    model = ft.load_model(str(MODEL_FILEPATH))


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
    text = text.lower()
    return text


def _clean_texts(data, text_key=TEXT_KEY):
    return [change_dict(row, text_key, _clean_text(row[text_key])) for row in
            data]


def _load_data(filename, text_key=TEXT_KEY, remove_empty=True):
    data = read_csv(filename)

    # add other conditions if needed
    # currently only ignores rows with no utterance

    if remove_empty:
        data = [row for row in data if len(row[text_key]) > 0]

    return data


def _write_ft_file(filename, text_key=TEXT_KEY, label_key=LABEL_KEY,
                   remove_empty=True, clean=True, relevant_labels=None,
                   prefix=PREFIX):
    data = _load_data(filename, text_key=text_key, remove_empty=remove_empty)

    if clean:
        data = _clean_texts(data, text_key=text_key)
        data = _clean_labels(data, label_key=label_key,
                             relevant_labels=relevant_labels)

    def add_prefix(label):
        return f"{prefix}{label if len(label) > 0 else 'unknown'}"

    write_lines(
        [' '.join([add_prefix(row[label_key]), row[text_key]]) for row in data],
        TRAIN_FT_FILEPATH
    )


def train(filename, text_key=TEXT_KEY, label_key=LABEL_KEY, remove_empty=True,
          clean=True, relevant_labels=None, prefix=PREFIX, multilabel=False):
    _write_ft_file(filename, text_key=text_key, label_key=label_key,
                   remove_empty=remove_empty, clean=clean,
                   relevant_labels=relevant_labels, prefix=prefix)

    global model

    # hyperparameters might need to be adjusted
    # https://fasttext.cc/docs/en/python-module.html
    model = ft.train_supervised(
        input=str(TRAIN_FT_FILEPATH),
        minCount=2,
        loss='ova' if multilabel else 'softmax'
    )

    MODEL_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILEPATH))


def predict(text):
    try:
        labels, probs = model.predict(_clean_text(text), k=-1)
        labels = [label.split(PREFIX)[1] for label in labels]
        return dict(zip(labels, probs))
    except AttributeError:
        raise TrainedModelNotFound
