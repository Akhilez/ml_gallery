from gym_nine_mens_morris.envs.nine_mens_morris_env import NineMensMorrisEnv, Pix
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import copy
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


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

        self.features = nn.ModuleList(
            [SelfAttention(embed_dim) for _ in range(feature_depth)]
        )
        self.phase_1 = nn.ModuleList(
            [SelfAttention(embed_dim) for _ in range(phase_1_depth)]
        )
        self.phase_2 = nn.ModuleList(
            [SelfAttention(embed_dim) for _ in range(phase_2_depth)]
        )

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


model = A9Model(8, 4, 4, 4).double().to(device)


def subsample_legal_positions(probs, legal_pos):
    """
    probs: tensor of shape (24)
    legal_pos: list of tuples. Shape: (n, 3)
    """
    flattened_idx = [np.ravel_multi_index(pos, (3, 2, 4)) for pos in legal_pos]
    return probs[flattened_idx]


def sample_action(legal_actions, pos1, pos2, move, kill, is_phase_1, argmax=False):
    probs = []

    legal_pos = list(
        set([action[0] for action in legal_actions])
    )  # [(3, 2, 4), (3, 2, 4), ... ]
    pos_probs_ = pos1 if is_phase_1 else pos2
    # subsample pos_probs with legal actions
    pos_probs = subsample_legal_positions(pos_probs_, legal_pos)
    pos_idx = (
        int(pos_probs.argmax())
        if argmax
        else int(torch.multinomial(pos_probs, 1).squeeze())
    )  # 24
    pos = legal_pos[pos_idx]  # (3, 2, 4)
    probs.append(pos_probs[pos_idx])

    # [0, 1, 2, 3]
    legal_moves = list(
        set(
            [
                action[1]
                for action in legal_actions
                if tuple(action[0]) == tuple(pos) and action[1] is not None
            ]
        )
    )
    if len(legal_moves) != 0:
        move = move[legal_moves]
        move_idx = (
            int(move.argmax()) if argmax else int(torch.multinomial(move, 1).squeeze())
        )  # 4
        move_ = legal_moves[move_idx]  # 4
        probs.append(move[move_idx])
    else:
        move_ = None

    legal_kills = list(
        set(
            [
                tuple(action[2])
                for action in legal_actions
                if tuple(action[0]) == tuple(pos) and action[2] is not None
            ]
        )
    )
    if len(legal_kills) != 0:
        kill = subsample_legal_positions(kill, legal_kills)
        kill_idx = (
            int(kill.argmax()) if argmax else int(torch.multinomial(kill, 1).squeeze())
        )
        kill_ = legal_kills[kill_idx]  # (3, 2, 4)
        probs.append(kill[kill_idx])
    else:
        kill_ = None

    return (pos, move_, kill_), torch.mean(torch.stack(probs))


def reset_opponent_model(opponent, prev_models):
    prev_models = prev_models[-10:]
    prev_models.append(copy.deepcopy(model))
    opponent.model = prev_models[np.random.choice(len(prev_models), 1)[0]]
    opponent.model.eval()
    return prev_models


def play(player_1, player_2, render=False):
    env = NineMensMorrisEnv()
    env.reset()
    if render:
        env.render()

    info = {}
    is_done = False
    while not is_done:
        player = player_1 if env.player == Pix.W else player_2
        state, reward, is_done, info = env.step(player(env))
        if render:
            env.render()

    winner = info.get("winner")
    if winner:
        return 1 if winner == Pix.W.string else 2
    return 0


def random_player(env, legal_actions=None):
    legal_actions = (
        legal_actions if legal_actions is not None else env.get_legal_actions()
    )
    if len(legal_actions) == 0:
        env.swap_players()
        return (0, 0, 0), None, None
    random_idx = int(torch.randint(low=0, high=len(legal_actions), size=(1,))[0])
    random_action = legal_actions[random_idx]
    return random_action


class AIPlayer:
    def __init__(self, model):
        self.model = model

    def __call__(self, env, legal_actions=None):
        legal_actions = (
            legal_actions if legal_actions is not None else env.get_legal_actions()
        )
        if len(legal_actions) == 0:
            env.swap_players()
            return (0, 0, 0), None, None

        xs = create_state_batch([env])

        was_train = self.model.training

        self.model.eval()
        with torch.no_grad():
            yh_pos_1, yh_pos_2, yh_move, yh_kill = self.model(
                xs
            )  # yh shape: (batch, 9)

        if was_train:
            self.model.train()

        action, _ = sample_action(
            legal_actions,
            yh_pos_1[0],
            yh_pos_2[0],
            yh_move[0],
            yh_kill[0],
            env.is_phase_1(),
            argmax=False,
        )

        return action


def randomize_ai_player(stats):
    batch_size = len(stats)

    # First half are False and second half are True
    bools = torch.arange(batch_size) > (batch_size - 1) / 2

    # Randomly shuffle equal number of Falses and Trues
    bools = bools[torch.randperm(batch_size)]

    # Set player's piece to these random values
    for i in range(batch_size):
        stats[i].player = Pix.W if bools[i] else Pix.B


def create_state_batch(envs):
    xs = [convert_inputs(env.board, env.player) for env in envs]
    xs = torch.stack(xs).long().to(device)
    return xs


def is_all_done(stats):
    for stat in stats:
        if not stat.env.is_done:
            return False
    return True


def get_credits(t, gamma):
    credits = []
    prev_credit = 1
    for i in range(t):
        credits.append(prev_credit)
        prev_credit *= gamma
    return torch.tensor(list(reversed(credits))).double().to(device)


def get_returns(rewards, gamma):
    total_t = len(rewards)
    returns = []
    prev_return = 0
    for t in range(total_t):
        prev_return = rewards[total_t - t - 1] + (gamma * prev_return)
        returns.append(prev_return)
    return torch.tensor(list(reversed(returns))).double().to(device)


def get_loss(stats):
    loss = 0
    for i_env in range(len(stats)):
        returns = get_returns(stats[i_env].rewards, gamma=0.99)
        probs = torch.log(torch.stack([prob for prob in stats[i_env].probs]))
        credits = get_credits(len(stats[i_env].rewards), gamma=0.99)

        loss += torch.mean(probs * credits * returns)
    return -1 * loss / len(stats)


class Stat:
    def __init__(self):
        self.player = Pix.W
        self.env = NineMensMorrisEnv()
        self.steps = []
        self.has_won = None
        self.probs = []
        self.rewards = []

        self.env.reset()


class EpisodicStat:
    def __init__(self, batch_size):
        self.loss = None
        self.stats = [Stat() for _ in range(batch_size)]
        self.time_step = 0


def plot_interval(stats, episode_number):
    losses = [stat.loss for stat in stats]

    print(f"{episode_number}: {np.mean(losses)}", end="\t")

    wins, loses, plays = 0, 0, 0
    for stat_ep in stats:
        for stat_t in stat_ep.stats:
            plays += 1
            if stat_t.has_won:
                wins += 1
            else:
                loses += 1

    print(f"W: {wins / plays * 100} L: {loses / plays * 100} P: {plays}")

    plt.plot(losses)
    plt.show()


def run_time_step(stat_ep, opponent):
    stats = stat_ep.stats
    xs = create_state_batch([stat.env for stat in stats])

    yh = model(xs)
    with torch.no_grad():
        yh_op = opponent.model(xs)

    for i in range(len(stats)):
        if not stats[i].env.is_done:
            env = stats[i].env

            legal_actions = env.get_legal_actions()
            if len(legal_actions) == 0:
                env.swap_players()
                continue

            # Is current player AI or other?
            if env.player == stats[i].player:
                action, prob = sample_action(
                    legal_actions,
                    yh[0][i],
                    yh[1][i],
                    yh[2][i],
                    yh[3][i],
                    env.is_phase_1(),
                )
                state, reward, is_done, info = env.step(action)

                stats[i].probs.append(prob)
                stats[i].rewards.append(reward)
            else:
                action, _ = sample_action(
                    legal_actions,
                    yh_op[0][i],
                    yh_op[1][i],
                    yh_op[2][i],
                    yh_op[3][i],
                    env.is_phase_1(),
                )
                _, _, is_done, info = env.step(action)

            if is_done:
                stats[i].has_won = stats[i].player.string == info.get("winner")

    stat_ep.time_step += 1


def train():
    batch_size = 4
    episodes = 10
    reset_length = 2
    episodic_stats = []
    prev_models = []

    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    opponent = AIPlayer(copy.deepcopy(model))
    opponent.model.eval()

    for episode in range(episodes):
        stat_ep = EpisodicStat(batch_size)

        randomize_ai_player(stat_ep.stats)

        # Monte Carlo loop
        while not is_all_done(stat_ep.stats) and stat_ep.time_step < 1000:
            run_time_step(stat_ep, opponent)

        loss = get_loss(stat_ep.stats)

        optim.zero_grad()
        loss.backward()
        optim.step()

        stat_ep.loss = loss.item()
        episodic_stats.append(stat_ep)

        if (episode + 1) % reset_length == 0:
            plot_interval(episodic_stats, episode)
            episodic_stats = []
            prev_models = reset_opponent_model(opponent, prev_models)
