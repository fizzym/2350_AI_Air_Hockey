import gymnasium as gym
from air_hockey_gym.envs import AirHockeyEnv
import time

env = AirHockeyEnv(render_mode="human")

for i in range(0,1000):
	env.render()
	env.step([])



env.close()