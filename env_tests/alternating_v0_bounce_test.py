import gymnasium as gym
from air_hockey_gym.envs.single_mallet_alternating_v0 import SingleMalletAlternatingEnv
import time
import numpy as np

NUM_EPS = 100
#Simple script to visually observe bounce shots in alternating environment
#Also prints number of shots that missed goal

max_rew = 1.0
#Spawn mallet 1 in corner and puck around center of opponent's side
env = SingleMalletAlternatingEnv(max_reward=max_rew, render_mode="human",
                                 mal1_box_def=[[-0.95,0.45],[-0.95,0.45]],  
                                 puck_box_def=[(0.10,0.25), (0.55,-0.25)])

obs = env.reset_bounce_def(True)

resets = 0
num_misses = 0

while resets < 100:
    ob, rew, term, _, info, = env.step(0)

    if(term):
        resets += 1
        if info["termination_reason"] == "Max steps reached":
            num_misses += 1
        
        env.num_steps = 0
        env.reset_bounce_def(resets % 2)

print("Missed", num_misses, "shots")

env.close()
