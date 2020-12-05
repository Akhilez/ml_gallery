from app.nlp.next_char.utils import save_vocab
from mlg.settings import BASE_DIR
from collections import Counter
from torchtext.vocab import Vocab
import io

cleaned_file_path = f'{BASE_DIR}/data/subtitles/cleaned.txt'
sequence_file_path = f'{BASE_DIR}/data/subtitles/sequences.txt'
vocab_file_path = f'{BASE_DIR}/data/subtitles/char_vocab.txt'

seq_len = 25

pad_tkn = '~'
unk_tkn = '*'
eos_tkn = '\n'

counter = Counter()
with io.open(cleaned_file_path, encoding="utf8") as f:
    for string_ in f:
        counter.update(list(string_))

vocab = Vocab(counter, specials=[pad_tkn, unk_tkn, eos_tkn])
print(vocab.itos)
print(len(vocab.itos))
print(counter)

with open(sequence_file_path, 'w') as op_file:
    with io.open(cleaned_file_path, encoding="utf8") as f:
        for string_ in f:
            seq = []
            for i in range(min(seq_len, len(string_))):
                # seq.append(string_[i])
                seq.append(str(vocab[string_[i]]))

                seq_ = [pad_tkn] * (seq_len - i - 1) + seq

                op_file.write(','.join(seq_) + '\n')

print("Done writing to output")


save_vocab(vocab, vocab_file_path)
print(f"saved vocab to {vocab_file_path}")
