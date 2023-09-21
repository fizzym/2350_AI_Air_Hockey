#!/usr/bin/env python3

"""

### Cross-Entropy Numpy
Uses Numpy matrix to hold weights and biases and adjusts weights using Cross-Entropy

"""

from air_hockey_gym.envs import SingleMalletBlockEnv
import numpy as np
import pandas as pd

max_rew = 20.0
NUM_STEPS = 200
NUM_ITERATIONS = 1000
NUM_WEIGHTS = 24
ELITE_FRACTION = 1/4
LOAD = True
# env = AirHockeyEnv(max_reward=max_rew, render_mode="human")
env = SingleMalletBlockEnv(max_reward=max_rew, render_mode="human", puck_box=[(0.25,0),(0.25,0)], mal2_max_y_offset=0, mal2_vel_range=[1,1])

#File to test proper functioning of AirHockeyEnv
#Tests start position is correct and that env will terminate correctly when puck reaches both goals
#After tests, both mallets should move randomly around table

def get_action(weights, observations):
    return (np.matmul(weights[0:2,0:12], observations.flatten()) + weights[0:2,12]).reshape(-1)

best_weights = np.zeros((2, 13))
if LOAD:
    # best_weights = pd.read_csv("./best_weights_2.csv")
    best_weights = np.loadtxt('./best_weights_2.csv', delimiter=',')
std = np.ones((2, 13))

for i in range(NUM_ITERATIONS):
    print(f"Iteration {i}: {best_weights}")

    weights_pop = [best_weights + np.multiply(std,np.random.randn(2, 13)) for i_weight in range(NUM_WEIGHTS)]
    rewards = np.zeros(NUM_WEIGHTS)

    for j in range(NUM_WEIGHTS):
        weights_1 = weights_pop[j].reshape(2,13)

        obs, _ = env.reset()

        for k in range(NUM_STEPS):
            action_1 = get_action(weights_1, obs)
            action = np.concatenate((action_1, action_1))

            obs, rew, term, _, _, = env.step(action)

            rewards[j] += rew

            if(term):
                break

        print(f"Weight {j}: {rewards[j]}")

    elite_idxs_1 = np.array(rewards).argsort()[-int(ELITE_FRACTION*NUM_WEIGHTS):]
    elite_weights_1 = [weights_pop[idx] for idx in elite_idxs_1]

    for x in range(2):
        for y in range(13):
            arr = []
            for weights in elite_weights_1:
                arr.append(weights[x,y])
            best_weights[x,y] = np.array(arr).mean()
            std[x,y] = max(np.array(arr).std(), 0.25)

    # best_weights = np.array(elite_weights_1).mean()
    # std = np.array(elite_weights_1).std()

    pd.DataFrame(best_weights).to_csv("./best_weights.csv", header=None, index=None)

env.close()
