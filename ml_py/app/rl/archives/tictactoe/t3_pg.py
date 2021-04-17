import matplotlib.pyplot as plt
import copy
from gym_tic_tac_toe.envs.tic_tac_toe_env import TicTacToeEnv, Pix
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    def __init__(self, embed_dim, attentions_depth=2):
        # type: (int, int) -> None
        super().__init__()
        self.piece_embed = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)
        self.pos_embed = nn.Embedding(num_embeddings=9, embedding_dim=embed_dim)

        self.squeeze = nn.Linear(2 * embed_dim, embed_dim)

        self.attentions = nn.ModuleList(
            [SelfAttention(embed_dim) for _ in range(attentions_depth)]
        )

        self.out = nn.Linear(9 * embed_dim, 9)

    def forward(self, x):
        """
        x: int tensor of shape (batch_size, seq_len)
        """
        # Positional Embeddings
        batch_size = len(x)
        pos = torch.arange(9).unsqueeze(0).expand((batch_size, 9)).T.to(device)
        pos_embed = self.pos_embed(pos)  # shape: (seq, batch, embed)

        # Concatenate positional embeddings
        x_embed = self.piece_embed(x.flatten(1).T)  # shape: (seq, batch, embed)
        x = torch.cat((x_embed, pos_embed), 2)  # shape: (seq, batch, 2 * embed)

        # Squeeze to embed_dim
        x = F.leaky_relu(self.squeeze(x))  # shape: (seq, batch, embed)
        x = F.dropout(x, 0.3)

        # Attention layers
        for attention in self.attentions:
            x = F.leaky_relu(attention(x)[0])
            x = F.dropout(x, 0.3)

        # Flatten the sequence
        x = x.transpose(0, 1).flatten(1)

        # Output action
        x = self.out(x)
        return x


model = T3Model(4, 1).double().to(device)


default_action = 4
player_p = Pix.X
player_n = Pix.O
gamma_returns = 0.75
gamma_credits = 0.75
all_actions = torch.arange(0, 9)


class CustomPreSampler:
    def __init__(self):
        self.episodes = 0

    def __call__(self, probs):
        """
        1. Add dirchilet noise
        2. Set temperature based on episode number or given temperature
        """
        episode_number = self.episodes

        # tau = max((1 / np.log(episode_number)) * 5, 0.7)
        probs = F.gumbel_softmax(probs, tau=1, dim=0)

        # Add Noise
        # noise = torch.rand(len(probs)) * 0.01
        # probs = probs + noise

        # if episode_number % 145 == 0:
        # print('-------- LEGAL PROBS: ')
        # print(probs)

        return probs


# Game specific
def sample_action(all_actions, legal_action_idx, probs, pre_sampling_fn=None):
    """
    Return action, its probability
    1. Get all legal actions.
    2. Apply pre-sampling function
    3. Get all probs of legal actions
    4. Sample action idx from probs
    5. Return action and prob of the action idx
    """
    legal_actions = all_actions[legal_action_idx]
    legal_probs = probs[legal_action_idx]
    if pre_sampling_fn:
        legal_probs = pre_sampling_fn(legal_probs)
    else:
        legal_probs = torch.softmax(legal_probs, 0)
    idx = torch.multinomial(legal_probs, 1)[0]
    return legal_actions[idx], legal_probs[idx]


def reset_opponent_model(opponent, prev_models):
    prev_models = prev_models[-20:]
    prev_models.append(copy.deepcopy(model))
    opponent.model = prev_models[np.random.choice(len(prev_models), 1)[0]]
    opponent.model.eval()
    return prev_models


def play(player_1, player_2, render=False):
    env = TicTacToeEnv()
    env.reset()
    if render:
        env.render()

    info = {}
    is_done = False
    while not is_done:
        player = player_1 if env.player == player_p else player_2
        state, reward, is_done, info = env.step(player(env))
        if render:
            env.render()

    winner = info.get("winner")
    if winner:
        return 1 if winner == player_p.string else 2
    return 0


def random_player(env, legal_actions=None):
    legal_actions = (
        legal_actions if legal_actions is not None else env.get_legal_actions()
    )
    if len(legal_actions) == 0:
        env.swap_players()
        return default_action
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
            return default_action

        xs = create_state_batch([env])

        was_train = self.model.training

        self.model.eval()
        with torch.no_grad():
            probs = self.model(xs)  # yh shape: (batch, 9)

        if was_train:
            self.model.train()

        action, _ = sample_action(
            all_actions=all_actions, legal_action_idx=legal_actions, probs=probs[0]
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
        stats[i].player = Pix.X if bools[i] else Pix.O


def create_state_batch(envs):
    xs = [convert_inputs(env.state, env.player) for env in envs]
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
        returns = get_returns(stats[i_env].rewards, gamma=gamma_returns)
        probs = torch.log(torch.stack([prob for prob in stats[i_env].probs]))
        credits = get_credits(len(stats[i_env].rewards), gamma=gamma_credits)

        loss += torch.mean(probs * credits * returns)
    return -1 * loss / len(stats)


class Stat:
    def __init__(self):
        self.player = player_p
        self.env = TicTacToeEnv()
        self.steps = []
        self.has_won = None
        self.has_drawn = False
        self.probs = []
        self.rewards = []

        self.env.reset()


class EpisodicStat:
    def __init__(self, batch_size):
        self.loss = None
        self.stats = [Stat() for _ in range(batch_size)]


def plot_interval(stats, episode_number, plot=False):
    losses = [stat.loss for stat in stats]

    print(f"{episode_number}: {np.mean(losses)}", end="\t")

    wins, loses, plays = 0, 0, 0
    for stat_ep in stats:
        for stat_t in stat_ep.stats:
            plays += 1
            if stat_t.has_won:
                wins += 1
            elif not stat_t.has_drawn:
                loses += 1

    print(f"W: {wins / plays * 100} L: {loses / plays * 100} P: {plays}")

    if plot:
        plt.plot(losses)
        plt.show()


def run_time_step(stats, opponent, sampler, episode_num):
    xs = create_state_batch([stat.env for stat in stats])

    yh = model(xs)
    with torch.no_grad():
        yh_op = opponent.model(xs)

    # if episode_num % 145 == 0:
    # print("All probs ------------------------")
    # print(yh[:5])

    for i in range(len(stats)):
        if not stats[i].env.is_done:
            env = stats[i].env

            legal_actions = env.get_legal_actions()
            if len(legal_actions) == 0:
                env.swap_players()
                continue

            # Is current player AI or other?
            if env.player == stats[i].player:
                action, prob = sample_action(all_actions, legal_actions, yh[i], sampler)
                state, reward, is_done, info = env.step(action)

                stats[i].probs.append(prob)
                stats[i].rewards.append(reward)
            else:
                action, _ = sample_action(all_actions, legal_actions, yh_op[i], sampler)
                _, _, is_done, info = env.step(action)

            if is_done:
                stats[i].has_won = stats[i].player.string == info.get("winner")
                stats[i].has_drawn = info.get("winner") is None


def train():
    batch_size = 64
    episodes = 1000
    reset_length = 50
    episodic_stats = []
    prev_models = []

    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler = CustomPreSampler()
    opponent = AIPlayer(copy.deepcopy(model))
    opponent.model.eval()

    for episode in range(episodes):
        stat_ep = EpisodicStat(batch_size)

        randomize_ai_player(stat_ep.stats)

        # Monte Carlo loop
        while not is_all_done(stat_ep.stats):
            run_time_step(stat_ep.stats, opponent, sampler, episode)

        loss = get_loss(stat_ep.stats)

        optim.zero_grad()
        loss.backward()
        optim.step()

        stat_ep.loss = loss.item()
        episodic_stats.append(stat_ep)
        sampler.episodes += 1
        print(".", end="")

        if (episode + 1) % reset_length == 0:
            plot_interval(episodic_stats, episode)
            episodic_stats = []
            prev_models = reset_opponent_model(opponent, prev_models)


train()

plays = [play(AIPlayer(model), random_player) for _ in range(100)]
winners = torch.tensor(plays).float()

draws = len(torch.nonzero(winners == 0))
wins = len(torch.nonzero(winners == 1))
loses = len(torch.nonzero(winners == 2))

print(draws, wins, loses)

plays = [play(random_player, AIPlayer(model)) for _ in range(100)]
winners = torch.tensor(plays).float()

draws = len(torch.nonzero(winners == 0))
wins = len(torch.nonzero(winners == 1))
loses = len(torch.nonzero(winners == 2))

print(draws, wins, loses)

play(random_player, AIPlayer(model), render=True)
play(AIPlayer(model), random_player, render=True)
