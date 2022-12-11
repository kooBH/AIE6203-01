import numpy as np
import psutil
import torch
from torch.nn import functional as F
from scipy.signal import convolve, gaussian

########  from utils.py ########

def get_cum_discounted_rewards(rewards, gamma):
    """
    evaluates cumulative discounted rewards:
    r_t + gamma * r_{t+1} + gamma^2 * r_{t_2} + ...
    """
    cum_rewards = []
    cum_rewards.append(rewards[-1])
    for r in reversed(rewards[:-1]):
        cum_rewards.insert(0, r + gamma * cum_rewards[0])
    return cum_rewards


def play_and_log_episode(env, agent, gamma=0.99, t_max=10000):
    """
    always greedy
    """
    states = []
    v_mc = []
    v_agent = []
    q_spreads = []
    td_errors = []
    rewards = []

    s = env.reset()
    for step in range(t_max):
        states.append(s)
        qvalues = agent.get_qvalues([s])
        max_q_value, min_q_value = np.max(qvalues), np.min(qvalues)
        v_agent.append(max_q_value)
        q_spreads.append(max_q_value - min_q_value)
        if step > 0:
            td_errors.append(
                np.abs(rewards[-1] + gamma * v_agent[-1] - v_agent[-2]))

        action = qvalues.argmax(axis=-1)[0]

        s, r, done, _ = env.step(action)
        rewards.append(r)
        if done:
            break
    td_errors.append(np.abs(rewards[-1] + gamma * v_agent[-1] - v_agent[-2]))

    v_mc = get_cum_discounted_rewards(rewards, gamma)

    return_pack = {
        'states': np.array(states),
        'v_mc': np.array(v_mc),
        'v_agent': np.array(v_agent),
        'q_spreads': np.array(q_spreads),
        'td_errors': np.array(td_errors),
        'rewards': np.array(rewards),
        'episode_finished': np.array(done)
    }

    return return_pack


def img_by_obs(obs, state_dim):
    """
    Unwraps obs by channels.
    observation is of shape [c, h=w, w=h]
    """
    return obs.reshape([-1, state_dim[2]])


def is_enough_ram(min_available_gb=0.1):
    mem = psutil.virtual_memory()
    return mem.available >= min_available_gb * (1024 ** 3)


def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) +
            final_val * cur_step) / total_steps


def step_decay(init_val, final_val, cur_step, total_steps,
               replay_memory_start_size=10 ** 5, eps_annealing_frames=500000,
               max_frames=2500000):
    slope = -(init_val - final_val) / eps_annealing_frames
    intercept = init_val - slope * replay_memory_start_size
    slope_2 = -(final_val - total_steps) / (max_frames - eps_annealing_frames - replay_memory_start_size)
    intercept_2 = total_steps - slope_2 * max_frames

    if cur_step < replay_memory_start_size:
        eps = init_val
    elif cur_step >= replay_memory_start_size and cur_step < replay_memory_start_size + eps_annealing_frames:
        eps = slope * cur_step + self.intercept
    elif cur_step >= replay_memory_start_size + eps_annealing_frames:
        eps = slope_2 * cur_step + self.intercept_2
    return eps


def smoothen(values):
    kernel = gaussian(100, std=100)
    # kernel = np.concatenate([np.arange(100), np.arange(99, -1, -1)])
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, 'valid')


########### from replay_buffer.py ##############

# This code is modified on 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np


class ReplayBuffer(object):
    def __init__(self, size, priority_replay=False, alpha=0.7, beta=0.5, eps=1e-7):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = {"obses_t": [], "actions": [], "rewards": [], "obses_tp1": [], "dones": []}
        self._maxsize = size
        self._next_idx = 0
        self._probabilities = []
        self.priority_replay = priority_replay
        self._eps = eps
        self.alpha = alpha
        self.beta = beta
        self._size = 0

    def __len__(self):
        return self._size

    def _max_priority(self):
        return np.max(self._probabilities) if self.priority_replay else 1.0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = {"obses_t": np.array([obs_t]),
                "actions": np.array([action]),
                "rewards": np.array([reward]),
                "obses_tp1": np.array([obs_tp1]),
                "dones": np.array([done])
                }

        if len(self) == 0:
            self._probabilities = np.zeros((self._maxsize), dtype=np.float32)
            self._probabilities[0] = 1.0
            for k in data.keys():
                self._storage[k] = np.zeros((self._maxsize, *data[k].shape[1:]), dtype=data[k].dtype)

        self._probabilities[self._next_idx] = self._max_priority()
        for k in data.keys():
            self._storage[k][self._next_idx] = data[k]

        self._size = min(self._size + 1, self._maxsize)
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def update_priority(self, td_loss):
        if self.priority_replay:
            self._probabilities[self.idxes] = np.power(np.abs(td_loss) + self._eps, self.alpha)

    def _encode_sample(self, idxes):

        return (
            self._storage["obses_t"][idxes],
            self._storage["actions"][idxes],
            self._storage["rewards"][idxes],
            self._storage["obses_tp1"][idxes],
            self._storage["dones"][idxes],
        )

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        probabilities = self._probabilities[:len(self)] / np.sum(self._probabilities[:len(self)])

        self.idxes = np.random.choice(
            range(len(self)),
            batch_size,
            p=probabilities,
        )
        if self.priority_replay:
            is_weight = np.power(len(self) * probabilities[self.idxes], -self.beta)
            is_weight /= is_weight.max()
        else:
            is_weight = np.ones(len(self.idxes))
        return (*self._encode_sample(self.idxes), is_weight)


######## from helper.py ########

def evaluate(env, agent, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    reward = 0

    n_lives = max(env.unwrapped.ale.lives(), 1)

    for _ in range(n_lives):
        s = env.reset()
        for _ in range(t_max):
            qvalues = agent.get_qvalues(s)
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _,_ = env.step(action)
            reward += r

            if done:
                break

    rewards.append(reward)
    return np.mean(rewards)


def evaluate_A3C(env, agent, n_games=1):
    """Plays an a game from start till done, returns per-game rewards """

    game_rewards = []
    n_lives = max(env.unwrapped.ale.lives(), 1)

    for _ in range(n_games):
        for i in range(n_lives):
            state = env.reset()
            total_reward = 0
            while True:
                # action = agent.sample_actions(agent([state]))[0]
                agent_outputs, _ = agent([state])
                action = agent.best_actions(agent_outputs)[0]
                state, reward, done, info = env.step(action)
                total_reward += reward
                # if reward !=0:
                # 	print(reward)
                if done:
                    break

        game_rewards.append(total_reward)
    return np.mean(game_rewards)


def evaluate_A3C_lstm(env, agent, n_games=1):
    """Plays an a game from start till done, returns per-game rewards """

    game_rewards = []
    for _ in range(n_games):
        state = env.reset()
        hidden_unit = None
        total_reward = 0
        while True:
            agent_outputs, hidden_unit = agent([state], hidden_unit)
            action = agent.best_actions(agent_outputs)[0]
            state, reward, done, info = env.step(action)
            total_reward += reward
            # if reward !=0:
            # 	print(reward)
            if done:
                break

        game_rewards.append(total_reward)
    return np.mean(game_rewards)


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0
    # env.reset()

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        qvalues = agent.get_qvalues(s)
        action = agent.sample_actions(qvalues)[0]
        s_n, r, done, truncated,info = env.step(action)

        exp_replay.add(s, action, r, s_n, done)
        sum_rewards += r
        s = s_n
        if done:
            s = env.reset()

    return sum_rewards, s


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network, is_weight,
                    gamma=0.99,
                    check_shapes=False,
                    device=torch.device('cpu'), double_dqn=True):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float)  # shape: [batch_size, *state_shape]

    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)  # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states

    predicted_qvalues = agent(states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues.gather(1, actions.view(-1, 1))

    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)

    if double_dqn:
        next_actions = agent(next_states).argmax(axis=-1)
    else:
        next_actions = target_network(next_states).argmax(axis=-1)

    # compute V*(next_states) using predicted next q-values
    next_state_values = predicted_next_qvalues.gather(1, next_actions.view(-1, 1))

    target_qvalues_for_actions = rewards.view(-1, 1) + is_not_done.view(-1, 1) * (gamma * next_state_values)

    error = torch.abs(predicted_qvalues_for_actions -
                      target_qvalues_for_actions.detach())

    # loss =torch.mean(torch.from_numpy(is_weight).to(device).detach()
    #             * torch.pow(predicted_qvalues_for_actions - target_qvalues_for_actions.detach(),2))

    loss = torch.mean(torch.from_numpy(is_weight).to(device).detach()
                      * F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach(),
                                         reduction='none'))

    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"

    return loss, error.detach().view(-1).cpu().numpy()

