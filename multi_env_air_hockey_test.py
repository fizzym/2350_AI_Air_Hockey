from air_hockey_gym.envs import AirHockeyEnv
import numpy as np
import time

from air_hockey_gym.envs.multi_env_wrapper import MultiEnvWrapper

max_rew = 5.0
# Placeholder constants, change during testing
SAMPLE_ACTION_STEPS = 100
NUM_ENVS = 10

envs = MultiEnvWrapper(AirHockeyEnv, NUM_ENVS, max_reward=max_rew, render_mode="rgb_array")

# File to test proper functioning of MultiEnvWrapper
# Currently just samples action in each environment
start = time.perf_counter()

obs = envs.reset()

for i in range(SAMPLE_ACTION_STEPS):
    actions = envs.action_space_sample()

    ret_val = envs.step(actions)

envs.close()

end = time.perf_counter()
print(f"Execution time of {NUM_ENVS} envs for {SAMPLE_ACTION_STEPS} steps: {end - start} s")
