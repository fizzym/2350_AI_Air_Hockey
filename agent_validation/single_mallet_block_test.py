import gymnasium as gym
from air_hockey_gym.envs import SingleMalletBlockDiscreteEnv
import time
import numpy as np

import torch
import torch.nn as nn

MAX_ACCEL = 2.5
HIDDEN_SIZE = 128
LOAD_PATH = "./multi_defence_model_40.pt"

class Net(nn.Module):
    '''
    @brief Takes an observation from the environment and outputs a probability 
           for each action we can take.
    '''
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

max_rew = 20.0
env = SingleMalletBlockDiscreteEnv(max_reward=max_rew, render_mode="human", mal2_vel_range = [0.5,1.5], max_accel=MAX_ACCEL)
obs_size = 12
n_actions = len(env.actions)

net = Net(obs_size, HIDDEN_SIZE, n_actions)
net.load_state_dict(torch.load(LOAD_PATH))
net.eval()
sm = nn.Softmax(dim=1)

obs, _ = env.reset()
obs = obs.flatten()

step = 0

while True:
    obs_v = torch.FloatTensor([np.array(obs)])
    act_probs_v = sm(net(obs_v))
    act_probs = act_probs_v.data.numpy()[0]
    action = np.random.choice(range(n_actions), p=act_probs)
    next_obs, reward, is_done, _, _ = env.step(action)
    next_obs = next_obs.flatten()
    
    step += 1
    if step == 200 or is_done:
        step = 0
        next_obs, _ = env.reset()
        next_obs = next_obs.flatten()
    obs = next_obs
