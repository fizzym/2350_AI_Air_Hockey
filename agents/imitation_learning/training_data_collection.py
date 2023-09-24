from air_hockey_gym.envs import SingleMalletBlockEnv
import time
import numpy as np
import pyautogui
from simple_pid import PID
import csv

'''
### Data Collection File
This file is used to collect observations and action data for the purpose of Imitation Learning.

When the simulation begins, the mouse cursor is used to control the player mallet. The goal is to play as well as possible against the
opponent and then save the observations and actions as a row vector. 

If a training session terminates with a negative reward, the entire session's data is discarded. Otherwise, it is saved.

The collected data can be used with Cross Entropy Agents, since it uses episodes to complete its training. The goal is to "seed" the
agent with high quality data so that it can learn desirable behaviours early.
'''

# Set data collection parameters
ACTION_STEPS = 100000
TESTING_DELAY = 3  # seconds
MAX_MALLET_ACCEL = 5  # mallet acceleration
SCREEN_X, SCREEN_Y = pyautogui.size()

# Make sure to change these to not override current data files
data_path = "agents/imitation_learning/data/"
# data_file = "training_no_variation.csv"
# data_file = "training_vel_variations1.csv"
data_file = "training_vel_pos_variations.csv"

# Transform a coordinate on screen to an Air Hockey table coordinate
def transform(point):
    new_x = min((2* point[0] / SCREEN_X) - 1, 0)
    new_y = -((point[1] / SCREEN_Y) - 0.5)
    return new_x, new_y

max_rew = 5.0
# env = SingleMalletBlockEnv(max_reward=max_rew, render_mode="human", puck_box=[(0.25,0),(0.25,0)], mal2_max_y_offset=0, mal2_vel_range=[1,1])
# env = SingleMalletBlockEnv(max_reward=max_rew, render_mode="human", puck_box=[(0.25,0),(0.25,0)], mal2_max_y_offset=0, mal2_vel_range=[0.5,2])
env = SingleMalletBlockEnv(max_reward=max_rew, render_mode="human")

pid_x = PID(15, 1, 2, setpoint=0)
pid_y = PID(15, 1, 2, setpoint=0)

env.render()
time.sleep(TESTING_DELAY)

obs, _ = env.reset()
data = []

with open(data_path + data_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["p_xpos", "p_ypos", "p_xvel", "p_yvel", "mal1_xpos", "mal1_ypos", "mal1_xvel", "mal1_yvel", "mal2_xpos", "mal2_ypos", "mal2_xvel", "mal2_yvel", "a_x", "a_y"])
    for i in range(ACTION_STEPS):
        # Retrieve mouse position and calculate target pos on table
        mouse_pos = pyautogui.position()
        target_pos = transform(mouse_pos)

        pid_x.setpoint = target_pos[0]
        pid_y.setpoint = target_pos[1]

        action = [pid_x(obs[1][0]), pid_y(obs[1][1])]

        obs, rew, term, _, _ = env.step(action)
        data.append(np.concatenate((obs.flatten(), action), axis=None))

        if term:
            # Only record data if it has positive reward
            if rew > 0:
                writer.writerows(data)
            else:
                print("Bad data, reject it")
            data.clear()

            env.reset()

env.close()
