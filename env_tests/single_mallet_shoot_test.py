import gymnasium as gym
from air_hockey_gym.envs import SingleMalletShootDiscreteEnv

SIMULATION_STEPS = 10000

max_rew = 5.0
env = SingleMalletShootDiscreteEnv(max_reward=max_rew, render_mode="human")

#File to test SingleMalletShootDiscreteEnv
#Currently just runs and resets environment continously for SIMULATION_STEPS steps with no actions

obs, _ = env.reset()
for i in range(SIMULATION_STEPS):
    obs, rew, term, _, _ = env.step(0)
    
    if term:
        env.reset()

env.close()
