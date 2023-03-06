import gymnasium as gym
from air_hockey_gym.envs import AirHockeyEnv
import time

env = AirHockeyEnv(render_mode="human")

action = [5,0]
print(env.action_space)
for i in range(0,1000):
	env.render()
	if i == 250:
		action = [-5,0]
	if i == 500:
		action = [0,5]
	if i == 750:
		action = [0,-5]
		
	env.step(action)



env.close()