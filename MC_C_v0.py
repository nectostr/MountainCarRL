import gym
import torch.nn as nn
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from core.buffer import ReplayBuffer
from core.random_process import OUNoise
from core.util import hard_update, soft_update, get_class_attr_val

##As an example of code was used #https://github.com/blackredscarf/pytorch-DDPG

class Config:
    env: str = None
    episodes: int = None
    max_steps: int = None
    max_buff: int = None
    batch_size: int = None
    state_dim: int = None
    state_high = None
    state_low = None
    seed = None

    output = 'out'

    action_dim: int = None
    action_high = None
    action_low = None
    action_lim = None

    learning_rate: float = None
    learning_rate_actor: float = None

    gamma: float = None
    tau: float = None
    epsilon: float = None
    eps_decay = None
    epsilon_min: float = None

    use_cuda: bool = True

    checkpoint: bool = False
    checkpoint_interval: int = None

    use_matplotlib: bool = False

    record: bool = False
    record_ep_interval: int = None

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, eps=0.03):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1 + action_dim, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, 1)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        s1 = self.relu(self.fc1(state))
        x = torch.cat((s1, action), dim=1)

        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, eps=0.03):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, action_dim)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x)) # tanh limit (-1, 1)
        return action


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class DDPG:
    def __init__(self, config):
        self.config = config
        self.init()

    def init(self):
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.is_training = True
        self.randomer = OUNoise(self.action_dim)
        self.buffer = ReplayBuffer(self.config.max_buff)

        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.config.learning_rate_actor)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.config.learning_rate)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        if self.config.use_cuda:
            self.cuda()

    def learning(self):
        s1, a1, r1, t1, s2 = self.buffer.sample_batch(self.batch_size)

        t1 = (t1 == False) * 1
        s1 = torch.tensor(s1, dtype=torch.float)
        a1 = torch.tensor(a1, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float)
        t1 = torch.tensor(t1, dtype=torch.float)
        s2 = torch.tensor(s2, dtype=torch.float)

        if self.config.use_cuda:
            s1 = s1.cuda()
            a1 = a1.cuda()
            r1 = r1.cuda()
            t1 = t1.cuda()
            s2 = s2.cuda()

        a2 = self.actor_target(s2).detach()
        target_q = self.critic_target(s2, a2).detach()
        y_expected = r1[:, None] + t1[:, None] * self.config.gamma * target_q
        y_predicted = self.critic.forward(s1, a1)

        critic_loss = nn.MSELoss()
        loss_critic = critic_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        pred_a = self.actor.forward(s1)
        loss_actor = (-self.critic.forward(s1, pred_a)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()


        soft_update(self.actor_target, self.actor, self.config.tau)
        soft_update(self.critic_target, self.critic, self.config.tau)

        return loss_actor.item(), loss_critic.item()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def decay_epsilon(self):
        self.epsilon -= self.config.eps_decay

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)


        if self.config.use_cuda:
            state = state.cuda()

        action = self.actor(state).detach()
        action = action.squeeze(0).cpu().numpy()
        action += self.is_training * max(self.epsilon, self.config.epsilon_min) * self.randomer.noise()
        action = np.clip(action, -1.0, 1.0)

        self.action = action
        return action

    def reset(self):
        self.randomer.reset()



class MountainCarContinuous:

    def run(self):
        env = gym.make("MountainCarContinuous-v0")
        # self.env.reset()
        # for _ in range(10):
        #     new_state, reward, done, _ = self.env.step(self.env.action_space.sample())
        #     print(new_state)
        config = Config()
        config.env = env
        config.gamma = 0.99
        config.episodes = 1000
        config.max_steps = 200
        config.batch_size = 128
        config.epsilon = 1.0
        config.eps_decay = 0.001
        config.max_buff = 1000000
        config.use_cuda = False
        config.learning_rate = 1e-3
        config.learning_rate_actor = 1e-4
        config.epsilon_min = 0.001
        config.epsilon = 1.0
        config.tau = 0.001
        config.action_dim = int(env.action_space.shape[0])
        config.action_lim = float(env.action_space.high[0])
        config.state_dim = int(env.observation_space.shape[0])

        agent = DDPG(config)
        agent.is_training = True
        rewards_by_target_updates = []
        total_step = 0

        for ep in range(0, config.episodes):
            state0 = env.reset()
            agent.reset()

            done = False
            step = 0
            actor_loss, critics_loss, reward = 0, 0, 0

            # # decay noise
            agent.decay_epsilon()

            while not done:
                action = agent.get_action(state0)

                state1, reward1, done, info = env.step(action)
                agent.buffer.add(state0, action, reward1, done, state1)
                state0 = state1

                if agent.buffer.size() > config.batch_size:
                    loss_a, loss_c = agent.learning()
                    actor_loss += loss_a
                    critics_loss += loss_c

                reward += reward1
                step += 1
                total_step += 1

                if step + 1 > config.max_steps:
                    break

            rewards_by_target_updates.append(reward)
            print(f'Epoch {ep}/{config.episodes}, reward {rewards_by_target_updates[-1]:.4f}')

        return rewards_by_target_updates

if __name__ == '__main__':
    MC = MountainCarContinuous()
    res = MC.run()
    plt.plot(res)
    plt.title("MountainCarContinuous-v0")
    plt.show()

