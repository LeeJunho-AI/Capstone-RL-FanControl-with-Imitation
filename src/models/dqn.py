import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from src.config.settings import *
from src.environment.trading_env import FanEnv

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_N)
        )
    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_SIZE):
        self.buffer = []
        self.capacity = capacity
    def push(self, s, a, r, s2, d):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s2, d))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

def dqn_train(q_init, episodes=EPISODES):
    q = q_init.to(DEVICE)
    q_target = QNet().to(DEVICE)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=LR)
    buffer = ReplayBuffer()
    env = FanEnv()
    rewards = []
    best_reward = float('-inf')
    for episode in range(episodes):
        s, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                q_values = q(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                a = q_values.argmax().item()
            s2, r, done, _, _ = env.step(a)
            buffer.push(s, a, r, s2, done)
            episode_reward += r
            s = s2
            if len(buffer) >= START_LEARN:
                batch = buffer.sample(BATCH)
                s_batch = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32, device=DEVICE)
                a_batch = torch.tensor([x[1] for x in batch], dtype=torch.long, device=DEVICE)
                r_batch = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=DEVICE)
                s2_batch = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32, device=DEVICE)
                d_batch = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    q2 = q_target(s2_batch).max(1)[0]
                    target = r_batch + GAMMA * q2 * (1 - d_batch)
                current = q(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
                loss = F.smooth_l1_loss(current, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if episode % TARGET_SYNC == 0:
                    q_target.load_state_dict(q.state_dict())
        rewards.append(episode_reward)
        if episode_reward > best_reward:
            best_reward = episode_reward
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Best Reward: {best_reward:.2f}")
    return q, np.array(rewards) 