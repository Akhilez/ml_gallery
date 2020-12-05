import csv


def save_vocab(vocab, path):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for token, index in vocab.stoi.items():
            writer.writerow([index, token])


def read_vocab(path):
    vocab = dict()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            index, token = line
            vocab[token] = int(index)
    return vocab
