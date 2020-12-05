import torch
from torchtext.data import Field, TabularDataset, BucketIterator
from mlg.settings import BASE_DIR
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cleaned_data_path = f'{BASE_DIR}/data/subtitles/cleaned_test.txt'
data_path = f'{BASE_DIR}/data/subtitles'

batch_size = 16
seq_len = 25

pad_tkn = '~'
unk_tkn = '*'
eos_tkn = '\n'
init_tkn = '>'

tokenize = lambda string: list(string)

TEXT = Field(sequential=True, tokenize=list, fix_length=seq_len, unk_token=unk_tkn, pad_first=False,
             pad_token=pad_tkn, eos_token=eos_tkn, init_token=init_tkn)

train_dataset, test_dataset = TabularDataset.splits(
    path=data_path,
    train='cleaned.txt', test='cleaned_test.txt',
    format='csv',
    skip_header=False,
    fields=[("text", TEXT)])

TEXT.build_vocab(train_dataset)

train_iter, test_iter = BucketIterator.splits(
    (train_dataset, test_dataset),
    batch_sizes=(batch_size, batch_size),
    device=device,
    sort_key=lambda txt: len(txt.text),
    sort_within_batch=False,
    repeat=True
)


for x in test_iter:
    print(x.text)
    break
