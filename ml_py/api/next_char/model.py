import os
import torch
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

models_path = f"./models"
vocab_file_path = f"{models_path}/char_vocab.pt"

pad_tkn = "~"
unk_tkn = "*"
eos_tkn = "\n"
init_tkn = ">"


class NextCharModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=self.embed_size
        )

        self.rnn = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size
            # nonlinearity='relu'
        )

        self.y = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, x):
        y = F.relu(self.embed(x))
        y, _ = self.rnn(y)
        return F.softmax(self.y(y), 2)


class NextChar:
    def __init__(self):
        self.vocab = torch.load(vocab_file_path)
        self.vocab_size = len(self.vocab.itos)
        self.model = self.load_model()

    def predict(self, sentence):
        length = len(sentence)
        terminal_chars = [eos_tkn, "\n", pad_tkn]
        max_len = 50
        next_char = 0
        self.model.eval()
        with torch.no_grad():
            while next_char not in terminal_chars and len(sentence) < max_len:
                seq = torch.tensor(
                    [
                        self.vocab[s] or self.vocab[unk_tkn]
                        for s in list(sentence.lower())
                    ],
                    device=device,
                    dtype=torch.long,
                ).view((-1, 1))
                preds = self.model(seq)
                m = int(preds[-1][0].argmax())
                next_char = self.vocab.itos[m]
                sentence = sentence + next_char
        return sentence[length:]

    def load_model(self, latest=True, name=None):
        model = NextCharModel(self.vocab_size, 512, 1024)
        try:
            if latest:
                name = max(os.listdir(models_path))
            model.load_state_dict(
                torch.load(f"{models_path}/{name}", map_location=torch.device(device))
            )
            print(f"Loading model {name}")
        except Exception as e:
            print(e)
        return model
