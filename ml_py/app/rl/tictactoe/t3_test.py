from gym_tic_tac_toe.envs.tic_tac_toe_env import TicTacToeEnv, Pix
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def convert_inputs(state, me):
    # type: (np.ndarray, [Pix.X, Pix.O]) -> torch.Tensor
    # state: ndarray of shape (batch, 3, 3, 3)
    # Converts all pieces of current player to 1, opponent to 2 and empty space to 0

    me, opponent = [1, 2] if me == Pix.X else [2, 1]

    state = state.argmax(2)

    state[state == me] = 10
    state[state == opponent] = 20

    return torch.tensor((state // 10).flatten()).to(device)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        # type: (int) -> None
        super().__init__()
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, 1, 0.3)

    def forward(self, x):
        """
        x: tensor of shape (batch, seq_len, embed_dim)
        """
        key = F.leaky_relu(self.key(x))  # Shape: (batch, seq, embed)
        value = F.leaky_relu(self.value(x))  # Shape: (batch, seq, embed)

        return self.attention(x, key, value)


class T3Model(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.piece_embed = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)
        self.pos_embed = nn.Embedding(num_embeddings=9, embedding_dim=embed_dim)

        self.linear1 = nn.Linear(2 * embed_dim, embed_dim)

        self.attention1 = SelfAttention(embed_dim)
        self.attention2 = SelfAttention(embed_dim)

        self.linear2 = nn.Linear(9 * embed_dim, 64)
        self.linear3 = nn.Linear(64, 9)

    def forward(self, x):
        """
        x: int tensor of shape (batch_size, seq_len)
        """
        batch_size = len(x)
        pos = torch.arange(9).unsqueeze(0).expand((batch_size, 9)).T
        pos_embed = self.pos_embed(pos)  # shape: (seq, batch, embed)

        x_embed = self.piece_embed(x.flatten(1).T)  # shape: (seq, batch, embed)

        x = torch.cat((x_embed, pos_embed), 2)  # shape: (seq, batch, 2 * embed)

        x = F.leaky_relu(self.linear1(x))  # shape: (seq, batch, embed)
        x = F.dropout(x, 0.3)

        x = F.leaky_relu(self.attention1(x)[0])
        x = F.dropout(x, 0.3)

        x = F.leaky_relu(self.attention2(x)[0])  # shape: (seq, batch, embed)
        x = F.dropout(x, 0.3)

        # Flatten the sequence
        x = x.transpose(0, 1).flatten(1)

        x = F.leaky_relu(self.linear2(x))
        x = F.dropout(x, 0.3)

        x = F.relu(self.linear3(x))
        return x


def train():
    env = TicTacToeEnv()
    env.reset()
    env.step(0)
    env.step(1)
    env.step(2)
    env.step(5)
    env.render()

    model = Model(8).double().to(device)

    x = convert_inputs(env.state, env.player).flatten()
    x = torch.stack((x, x))

    yh = model(x)
