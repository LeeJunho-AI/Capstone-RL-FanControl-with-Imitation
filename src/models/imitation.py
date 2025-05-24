import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.config.settings import *

def compute_desired_time(dist, wait, mn=0.3, mx=30.0):
    return np.clip(5.0 * dist + 1.0 * wait, mn, mx)

def time_to_action(desired_time, min_time=0.5, step=0.5):
    idx = int(round((desired_time - min_time) / step))
    return np.clip(idx, 0, ACTION_N - 1)

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_N)
        )
    def forward(self, x):
        return self.model(x)

def imitation_train():
    waits = np.linspace(0, 20, 100)
    dists = np.linspace(1, 5, 100)
    obs_list, action_list = [], []

    for wt in waits:
        for dist in dists:
            obs = np.array([wt / 20, dist / 5], np.float32)
            desired = compute_desired_time(dist, wt)
            act_idx = time_to_action(desired)
            obs_list.append(obs)
            action_list.append(act_idx)

    obs_arr = np.stack(obs_list)
    action_arr = np.array(action_list)

    model = MLPClassifier().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(1000):
        idx = np.random.choice(len(obs_arr), 256, replace=False)
        batch_obs = torch.tensor(obs_arr[idx], dtype=torch.float32, device=DEVICE)
        batch_action = torch.tensor(action_arr[idx], dtype=torch.long, device=DEVICE)
        preds = model(batch_obs)

        loss = loss_fn(preds, batch_action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model, losses 