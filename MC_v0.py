import gym
import torch.nn as nn
import torch
import copy
import random
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt



class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        """Сохраняет элемент в циклический буфер"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Возвращает случайную выборку указанного размера"""
        return list(zip(*random.sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)


class MountainCar:

    def create_new_model(self):
        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal(layer.weight)

        model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        target_model = copy.deepcopy(model)

        model.apply(init_weights)

        # Загружаем модель на устройство, определенное в самом начале (GPU или CPU)
        model.to(self.device)
        target_model.to(self.device)

        # Сразу зададим оптимизатор, с помощью которого будем обновлять веса модели
        optimizer = optim.Adam(model.parameters(), lr=0.00003)

        return model, target_model, optimizer

    def __init__(self, _device="cpu", _gamma=0.99):
        # self.env = gym.make("MountainCarContinuous-v0")
        # self.env.reset()
        self.device = torch.device(_device) #gpu
        self.gamma = _gamma

    def fit(self, batch, model, target_model, optimizer):
        state, action, reward, next_state, done = batch

        state = torch.tensor(state).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()
        reward = torch.tensor(reward).to(self.device).float()
        action = torch.tensor(action).to(self.device)
        done = torch.tensor(done).to(self.device)

        target_q = torch.zeros(reward.size()[0]).float().to(self.device)
        with torch.no_grad():
            target_q[done] = target_model(next_state).max(1)[0].detach()[done]
        target_q = reward + target_q * self.gamma

        q = model(state).gather(1, action.unsqueeze(1))

        loss = F.smooth_l1_loss(q, target_q.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def select_action(self, state, epsilon, model):
        if random.random() < epsilon:
            return random.randint(0, 2)
        return model(torch.tensor(state).to(self.device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()

    def run(self):
        env = gym.make("MountainCar-v0")
        # self.env.reset()
        # for _ in range(10):
        #     new_state, reward, done, _ = self.env.step(self.env.action_space.sample())
        #     print(new_state)
        target_update = 1000
        batch_size = 128
        max_steps = 100000
        max_epsilon = 0.5
        min_epsilon = 0.1
        memory = Memory(5000)
        model, target_model, optimizer = self.create_new_model()
        state = env.reset()
        rewards_by_target_updates = []
        for step in tqdm(range(max_steps)):
            epsilon = max_epsilon - (max_epsilon - min_epsilon) * step / max_steps
            action = self.select_action(state, epsilon, model)
            new_state, reward, done, _ = env.step(action)

            modified_reward = reward + 273 * (self.gamma * abs(new_state[1]) - abs(state[1]))

            memory.push((state, action, modified_reward, new_state, done))
            if done:
                state = env.reset()
                done = False
            else:
                state = new_state

            if step > batch_size:
                self.fit(memory.sample(batch_size), model, target_model, optimizer)

            if step % target_update == 0:
                target_model = copy.deepcopy(model)

                state = env.reset()
                total_reward = 0
                while not done:
                    action = self.select_action(state, 0, target_model)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward

                state = env.reset()
                rewards_by_target_updates.append(total_reward)
        return rewards_by_target_updates

if __name__ == '__main__':
    MC = MountainCar()
    res = MC.run()
    plt.plot(res)
    plt.title("MountainCar-v0")
    plt.show()
