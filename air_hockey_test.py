import gymnasium as gym
from air_hockey_gym.envs import AirHockeyEnv
import time

env = AirHockeyEnv(render_mode="human")

#File to test proper functioning of AirHockeyEnv
#Tests start position is correct and that env will terminate correctly when puck reaches both goals
#After tests, both mallets should move randomly around table


#Ensure starting location is correct

#Mallet 1 starts at (-0.25, 0), mallet 2 starts at (0.25, 0), puck at (0,0) all in world frame (from XML)
#All starting vel are zero
start_arr =[[0,0,0,0], [-0.25,0,0,0], [0.25,0,0,0]]
expect_start = {"mal1": start_arr, "mal2": start_arr}

obs = env.reset()
assert expect_start == obs


#Test termination in mallet 2 goal
action = [5,0, 0, 1]
terminated_correctly = False
for i in range(50):
	obs, rew, term, _, _ = env.step(action)
	env.render()
	if(term):
		terminated_correctly = True
		break

assert terminated_correctly

obs = env.reset()
assert expect_start == obs

#Test termination in mallet 1 goal
action = [0,1, 5, 0]
terminated_correctly = False
for i in range(50):
	obs, rew, term, _, _ = env.step(action)
	env.render()
	if(term):
		terminated_correctly = True
		break

assert terminated_correctly

obs = env.reset()
assert expect_start == obs

for i in range(0,500):
	
	action = env.action_space.sample()
	env.step(action)
	env.render()



env.close()