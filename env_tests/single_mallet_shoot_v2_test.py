from air_hockey_gym.envs import SingleMalletShootEnvV2
from numpy.random import uniform as unif
from numpy.random import randint

SIMULATION_STEPS = 1000
MAX_ACCEL = 5

max_rew = 5.0
env = SingleMalletShootEnvV2(max_reward=max_rew, render_mode="human")

# Tests discrete action space

obs, _ = env.reset()
for i in range(SIMULATION_STEPS):
    obs, rew, term, _, _ = env.step(randint(0,9))
    
    if term:
        env.reset()

env.close()

env = SingleMalletShootEnvV2(max_reward=max_rew, discrete_actions=False, render_mode="human")

# Tests a continous action space

obs, _ = env.reset()
for i in range(SIMULATION_STEPS):
    obs, rew, term, _, _ = env.step([unif(-MAX_ACCEL, MAX_ACCEL), unif(-MAX_ACCEL, MAX_ACCEL)])
    
    if term:
        env.reset()

env.close()
