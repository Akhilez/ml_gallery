from gym_nine_mens_morris.envs.nine_mens_morris_env import NineMensMorrisEnv, Pix
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def convert_inputs(state, me):
    # type: (np.ndarray, [Pix.W, Pix.B]) -> torch.Tensor
    # state: ndarray of shape (3, 2, 4, 4)
    # Converts all pieces of current player to 1, opponent to 2 and empty space to 0

    me, opponent = [1, 2] if me == Pix.W else [2, 1]

    state = state.argmax(3)

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


class A9Model(nn.Module):
    def __init__(self, embed_dim, feature_depth=2, phase_1_depth=2, phase_2_depth=2):
        # type: (int, int, int, int) -> None
        super().__init__()
        self.piece_embed = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)
        self.pos_embed = nn.Embedding(num_embeddings=3 * 2 * 4, embedding_dim=embed_dim)

        self.squeeze = nn.Linear(2 * embed_dim, embed_dim)

        self.features = nn.ModuleList([SelfAttention(embed_dim) for _ in range(feature_depth)])
        self.phase_1 = nn.ModuleList([SelfAttention(embed_dim) for _ in range(phase_1_depth)])
        self.phase_2 = nn.ModuleList([SelfAttention(embed_dim) for _ in range(phase_2_depth)])

        self.phase_1_pos = nn.Linear(3 * 2 * 4 * embed_dim, 3 * 2 * 4)
        self.phase_2_pos = nn.Linear(3 * 2 * 4 * embed_dim, 3 * 2 * 4)
        self.phase_2_kill = nn.Linear(3 * 2 * 4 * embed_dim, 3 * 2 * 4)
        self.phase_2_move = nn.Linear(3 * 2 * 4 * embed_dim, 4)

    def forward(self, x):
        """
        x: int tensor of shape (batch_size, seq_len)
        """
        # Positional Embeddings
        batch_size = len(x)
        pos = torch.arange(3 * 2 * 4).unsqueeze(0).expand((batch_size, 3 * 2 * 4)).T
        pos_embed = self.pos_embed(pos)  # shape: (seq, batch, embed)

        # Concatenate positional embeddings
        x_embed = self.piece_embed(x.flatten(1).T)  # shape: (seq, batch, embed)
        x = torch.cat((x_embed, pos_embed), 2)  # shape: (seq, batch, 2 * embed)

        # Squeeze to embed_dim
        x = F.leaky_relu(self.squeeze(x))  # shape: (seq, batch, embed)
        x = F.dropout(x, 0.3)

        # Feature layers
        for attention in self.features:
            x = F.leaky_relu(attention(x)[0])
            x = F.dropout(x, 0.3)

        # ======= Phase 1 =========

        # Features
        f1 = x
        for attention in self.phase_1:
            f1 = F.leaky_relu(attention(f1)[0])
            f1 = F.dropout(f1, 0.3)

        # Flatten the sequence
        f1 = f1.transpose(0, 1).flatten(1)

        # Output action
        f1 = torch.softmax(self.phase_1_pos(f1), 1)

        # ======= Phase 2 =========

        # Features
        f2 = x
        for attention in self.phase_2:
            f2 = F.leaky_relu(attention(f2)[0])
            f2 = F.dropout(f2, 0.3)

        # Flatten the sequence
        f2 = f2.transpose(0, 1).flatten(1)

        # Output action
        f2_pos = torch.softmax(self.phase_2_pos(f2), 1)

        # Output move
        move = torch.softmax(self.phase_2_move(f2), 1)

        # Output kill
        kill = torch.softmax(self.phase_2_kill(f2), 1)

        return f1, f2_pos, move, kill


env = NineMensMorrisEnv()
env.reset()
env.step((0, 0, 0))
env.step((0, 0, 1))
x = torch.stack([convert_inputs(env.board, Pix.W)])
print(x)

model = A9Model(4, 1, 1, 1)
f1, f2, move, kill = model(x)

print(f1.shape)
print(f2.shape)
print(kill.shape)
print(move)

