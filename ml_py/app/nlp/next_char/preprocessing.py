from app.nlp.next_char.utils import save_vocab
from mlg.settings import BASE_DIR
import os
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import io

input_dir_path = f"{BASE_DIR}/data/subtitles"
output_file_path = f"{BASE_DIR}/data/subtitles/tokens.txt"
vocab_file_path = f"{BASE_DIR}/data/subtitles/vocab.txt"
cleaned_file_path = f"{BASE_DIR}/data/subtitles/cleaned.txt"

file_names = os.listdir(input_dir_path)
file_paths = [f"{input_dir_path}/{file_name}" for file_name in file_names]

tokenizer = get_tokenizer("basic_english", language="en")

counter = Counter()
for file_path in file_paths:
    with io.open(file_path, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
        print(f"Done counting words of {file_path}")
vocab = Vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])

with open(cleaned_file_path, "w") as clean_file:
    with open(output_file_path, "w") as op_file:
        for file_path in file_paths:
            raw_iter = iter(io.open(file_path, encoding="utf8"))
            for raw in raw_iter:
                tokenized = tokenizer(raw)
                clean_file.write(" ".join(tokenized) + "\n")
                op_file.write(
                    ",".join([str(vocab[token]) for token in tokenized]) + "\n"
                )
            print(f"Done writing tokens of {file_path}")


save_vocab(vocab, vocab_file_path)
print(f"saved vocab to {vocab_file_path}")
