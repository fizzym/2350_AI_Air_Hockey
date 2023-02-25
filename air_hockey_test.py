import gymnasium as gym
#import air_hockey_gym
from air_hockey_gym.envs import AirHockeyEnv
import time

env = AirHockeyEnv(render_mode="human")

for i in range(0,100):
	env.render()
	time.sleep(0.1)
#env.render()
#gym.make("AirHockey-v0", render_mode="human")
#observation, info = env.reset()


env.close()