import numpy as np
import random
from src.config.settings import ACTION_N

class FanEnv:
    def __init__(self, n=5, max_rounds=3):
        self.n = n
        self.R = max_rounds
        self.action_space = type('', (), {})()  # Dummy object
        self.action_space.n = ACTION_N
        self.action_space.sample = lambda: random.randint(0, ACTION_N - 1)
        self.observation_space = type('', (), {})()
        self.observation_space.shape = (2,)

    def _bidi(self):
        order = np.argsort(self.person_angles)
        return list(order) + list(order[-2:0:-1])

    def _obs(self):
        i = self.seq[self.idx]
        return np.array([
            np.clip(self.wait[i]/10, 0, 1),
            self.person_distances[i]/5
        ], np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.person_angles = np.sort(np.random.uniform(0, 180, self.n))
        self.person_distances = np.random.uniform(1, 5, self.n)
        self.fan_angle = self.fan_init = np.random.uniform(0, 180)
        self.seq = self._bidi()
        self.idx = 0
        self.rounds = 0
        self.blow = np.zeros(self.n)
        self.wait = np.zeros(self.n)
        return self._obs(), {}

    def step(self, act: int):
        i = self.seq[self.idx]
        bt = (act + 1) * 0.5
        d = np.abs(self.person_angles - self.fan_init)
        md = max(np.min(d), 1.0)
        des = np.clip(d[i]/md + 2.0*self.wait[i], 0.3, 30.0)
        desired_idx = int(round((des - 0.5) / 0.5))
        desired_idx = np.clip(desired_idx, 0, ACTION_N - 1)
        diff = abs(bt - des)

        self.fan_angle += np.clip(self.person_angles[i] - self.fan_angle, -10, 10)
        self.blow[i] += bt
        self.wait += bt * 0.05
        self.wait[i] += bt * (1 - 0.05)
        self.wait[i] = 0

        alpha = 0.02  # Much smaller value
        reward = 2.0 - diff
        reward -= alpha * abs(bt - des)
        reward = max(reward, -1.0)

        self.idx += 1
        if self.idx == len(self.seq):
            self.idx = 0
            self.rounds += 1
        done = self.rounds >= self.R
        return self._obs(), reward, done, False, {} 