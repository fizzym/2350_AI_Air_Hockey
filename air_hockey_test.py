import gymnasium as gym
from air_hockey_gym.envs import AirHockeyEnv
import time

env = AirHockeyEnv(render_mode="human")

#File to test proper functioning of AirHockeyEnv
#Mallet should move forward to hit puck then move randomly around table

action = [5,0]
for i in range(0,500):
	
	if i > 25:
		action = env.action_space.sample()
	env.step(action)
	env.render()


env.close()