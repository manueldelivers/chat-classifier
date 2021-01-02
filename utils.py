import csv
import sys


def change_dict(d, key, value):
    d[key] = value
    return d


def read_csv(filename):
    # csv.field_size_limit(sys.maxsize)

    with open(filename, 'r') as f:
        rows = list(csv.reader(f, delimiter=','))

    header = rows[0]
    data = []
    for row in rows[1:]:
        data.append(dict(zip(header, row)))

    return data


def write_lines(obj, filename):
    with open(filename, 'a') as f:
        f.writelines('\n'.join(iter(obj)))
