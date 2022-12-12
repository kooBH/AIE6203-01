import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    elif type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

class DQN(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0,
    dueling=False,
    policy=False,  # PolicyNetwork for policy gradient
    value=False    # ValueNetwork for policy gradient
    ):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        """
            Dueling  : Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.
            
            idea : How a' is better than others actions ? 
                   A(s,a) = Q(s,a) - V(s)
            effect : by seprating V, A, the model can learn which states are valueable.
            
        """
        self.dueling = dueling
        self.policy = policy
        self.value = value

        cnt_mult = 0

        if dueling : 
            cnt_mult += 1
        if policy : 
            cnt_mult += 1
        if value : 
            cnt_mult += 1

        if cnt_mult > 1 :
            raise Exception("ERROR::Do only one thing | dueling {} | policy {} | value {}.".format(dueling,policy,value))
        print("Network:: dueling {} | policy {} | value {}.".format(dueling,policy,value))
        self.block_1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2),
            nn.ReLU(),
            #nn.BatchNorm2d(256),
        )

        if dueling :
            self.block_5 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 1024),
            )
            self.value_net =  nn.Sequential(
                nn.Linear(1024,1)
            )
            self.advantage = nn.Sequential(
                nn.Linear(1024,n_actions)
            )
        elif value : 
            self.block_5 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1),
            )
        # vanila, policy gradient
        else : 
            self.block_5 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, n_actions),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if policy : 
            self.block_5[-1].weight.data = normalized_columns_initializer(
                self.block_5[-1].weight.data, 0.1)
            self.block_5[-1].bias.data.fill_(0)

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        model_device = next(self.parameters()).device
        state_t = torch.tensor(state_t, device=model_device, dtype=torch.float) / 128.0 - 1.0

        x = self.block_1(state_t)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        if self.dueling : 
            x = self.block_5(x)
            v = self.value_net(x)
            a = self.advantage(x)
            a_avg = torch.mean(a, axis=1, keepdims=True)
            qvalues = v + a - a_avg
            return qvalues
        elif self.policy : 
            ret = self.block_5(x)
            return ret
        # vanila, value
        else : 
            ret = self.block_5(x)
            return ret

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        # ADD : batch dim for torch layers
        states = np.expand_dims(states,0)
        with torch.no_grad():
            qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, x):
        if not self.policy :
            """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
            epsilon = self.epsilon
            batch_size, n_actions = x.shape
            random_actions = np.random.choice(n_actions, size=batch_size)
            best_actions = x.argmax(axis=-1)

            should_explore = np.random.choice(
                [0, 1], batch_size, p=[1 - epsilon, epsilon])
            return np.where(should_explore, random_actions, best_actions)
        # policy graident
        else :
            probs = F.softmax(x, dim=1)
            # torch.multinomial : Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.
            action = probs.multinomial(num_samples=1).detach()
            return action

    def best_actions(self, logits):
        probs = F.softmax(logits, dim=1)
        action = probs.max(1, keepdim=True)[1].detach().cpu().numpy()
        return action




