import matplotlib.pyplot as plt
import numpy as np
import torch
from src.config.settings import *
from src.models.imitation import compute_desired_time

def plot_training_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, color='#2E86C1', linewidth=2)
    plt.title("Training Loss Over Time", fontsize=14, pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross Entropy Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.show()

def plot_dqn_rewards(rewards, window=50):
    best_rewards = np.maximum.accumulate(rewards)
    # Calculate moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        x_ma = np.arange(window-1, len(rewards))
    else:
        moving_avg = rewards
        x_ma = np.arange(len(rewards))
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='#2E86C1', linewidth=1, alpha=0.3, label='Reward (raw)')
    plt.plot(x_ma, moving_avg, color='#F39C12', linewidth=2, label=f'Reward (moving avg, {window})')
    plt.plot(best_rewards, color='#E74C3C', linewidth=2, linestyle='--', label='Best Reward')
    plt.title("DQN Training Rewards Over Time", fontsize=14, pad=15)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(fontsize=12)
    plt.show()

def test_policy(model, device=DEVICE, n_tests=8):
    fig, axs = plt.subplots(4, 2, figsize=(14, 18))
    axs = axs.flatten()
    maes = []
    for seed in range(n_tests):
        np.random.seed(seed)
        test_waits = np.random.uniform(0, 20, 20)
        test_dists = np.random.uniform(1, 5, 20)
        p, t, e = [], [], []
        for wt, dist in zip(test_waits, test_dists):
            obs = np.array([wt / 20, dist / 5], np.float32)
            with torch.no_grad():
                pred = model(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                a = pred.argmax().item()
            desired = compute_desired_time(dist, wt)
            predicted_time = (a + 1) * 0.5
            p.append(predicted_time)
            t.append(desired)
            e.append(abs(predicted_time - desired))
        maes.append(np.mean(e))
        ax = axs[seed]
        ax.plot(p, 'o-', color='#2E86C1', label='Predicted', linewidth=2, markersize=6)
        ax.plot(t, 'x--', color='#E74C3C', label='Desired', linewidth=2, markersize=6)
        ax.set_title(f'Test {seed+1} | MAE {maes[-1]:.2f}s', fontsize=12, pad=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()
    print(f"\n✅ Overall Average MAE: {np.mean(maes):.3f} seconds")

def plot_policy_3d(model, device=DEVICE):
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    wait_grid = np.linspace(0, 20, 50)
    dist_grid = np.linspace(1, 5, 50)
    wait_mesh, dist_mesh = np.meshgrid(wait_grid, dist_grid)
    q_values = np.zeros_like(wait_mesh)
    for i in range(len(wait_grid)):
        for j in range(len(dist_grid)):
            wt = wait_grid[i]
            dist = dist_grid[j]
            obs = np.array([wt / 20, dist / 5], np.float32)
            with torch.no_grad():
                pred = model(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                probs = torch.softmax(pred, dim=-1)
                wind_time = ((torch.arange(ACTION_N, device=device) + 1) * 0.5 * probs).sum().item()
                q_values[j, i] = wind_time
    surf = ax.plot_surface(wait_mesh, dist_mesh, q_values, cmap='viridis', edgecolor='none', alpha=0.8, antialiased=True)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('Waiting Time (s)', fontsize=12, labelpad=10)
    ax.set_ylabel('Distance (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Wind Time (s)', fontsize=12, labelpad=10)
    ax.set_title('Learned Q-Function Visualization', fontsize=14, pad=15)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.view_init(elev=30, azim=45)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Wind Time (s)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    plt.tight_layout()
    plt.show() 