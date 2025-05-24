import torch

# 하이퍼파라미터
ACTION_N = 60  # 또는 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRIO_BETA_START = 0.4
PRIO_BETA_END = 1.0
PRIO_BETA_FRAC = 0.8
START_LEARN = 2000
GAMMA = 0.99
TARGET_SYNC = 200
LR = 2e-4
BATCH = 128
REPLAY_SIZE = 50000
EPISODES = 3000 