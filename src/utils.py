import numpy as np
import psutil
import torch
from torch.nn import functional as F
from scipy.signal import convolve, gaussian

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
        if agent.policy : 
            logits = agent.forward([s])
            action = agent.sample_actions(logits)[0]
            action = action.detach().cpu().numpy()[0]
        else : 
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
                    criterion = F.smooth_l1_loss,
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

