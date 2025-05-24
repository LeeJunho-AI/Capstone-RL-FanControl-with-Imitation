from src.models.imitation import imitation_train
from src.models.dqn import dqn_train
from src.utils.visualization import plot_training_loss, plot_dqn_rewards, test_policy, plot_policy_3d
from src.config.settings import DEVICE
import numpy as np
import matplotlib.pyplot as plt

print(f"[INFO] Using device: {DEVICE}")

# 1. 모방학습
model, losses = imitation_train()
plot_training_loss(losses)

test_policy(model)
plot_policy_3d(model)

# 2. DQN 학습
q_net, rewards = dqn_train(model)
plot_dqn_rewards(rewards)

test_policy(q_net)
plot_policy_3d(q_net)