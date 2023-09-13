import gymnasium as gym
from air_hockey_gym.envs import SingleMalletBlockEnv
import time
import numpy as np

max_rew = 5.0
env = SingleMalletBlockEnv(max_reward=max_rew, render_mode="human")

#File to test SingleMalletBlockEnv
#Currently just runs and resets environment continously for 1000 steps with no actions

obs, _ = env.reset()
for i in range(1000):
    obs, rew, term, _, _ = env.step([0,0])
    
    if term:
        env.reset()

env.close()
