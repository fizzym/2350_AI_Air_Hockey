from air_hockey_gym.envs import SingleMalletShootEnvV2
import time
import numpy as np
import pyautogui
from simple_pid import PID
import csv
import datetime

'''
### Data Collection File For Agent Offensive Training
This file is used to collect observations and action data for the purpose of Imitation Learning.

When the simulation begins, the mouse cursor is used to control the player mallet. The goal is to play as well as possible against the
opponent and then save the observations and actions as a row vector. 

If a training session terminates with a negative reward, the entire session's data is discarded. Otherwise, it is saved.

The collected data can be used with Cross Entropy Agents, since it uses episodes to complete its training. The goal is to "seed" the
agent with high quality data so that it can learn desirable behaviours early.
'''

# Set data collection parameters
MAX_EPISODES = 10
TESTING_DELAY = 3  # seconds
MAX_MALLET_ACCEL = 5  # mallet acceleration
REW_THRESHOLD = 0  # Episode reward threshold
SCREEN_X, SCREEN_Y = pyautogui.size()

# Special camera angle for data collection
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2,
    "lookat": np.array((0, 0, 0.1)),
    "elevation": -90,
}

# Formatting file name and path
cur_datetime = datetime.datetime.now()
formatted_datetime = cur_datetime.strftime("%m-%d-%H-%M")

data_path = "agents/imitation_learning/data/"
data_file = "offensive_train_data_" + formatted_datetime + ".csv"

# Transform a coordinate on screen to an Air Hockey table coordinate
def transform(point):
    new_x = min((2 * point[0] / SCREEN_X) - 1, 0)
    new_y = -((point[1] / SCREEN_Y) - 0.5)
    return new_x, new_y

max_rew = 5.0
# Randomize the mallet and puck spawn positions across their entire playable area
env = SingleMalletShootEnvV2(max_reward=max_rew, puck_box=[(-0.95,0.45),(-0.05,-0.45)],
                              mal1_box= [(-0.95,0.45),(-0.05,-0.45)], discrete_actions=False, render_mode="human", default_camera_config=DEFAULT_CAMERA_CONFIG)

pid_x = PID(15, 1, 2, setpoint=0)
pid_y = PID(15, 1, 2, setpoint=0)

env.render()
time.sleep(TESTING_DELAY)

obs, _ = env.reset()
data = []
episode_count = 0

# Termination flag
end_script = False

with open(data_path + data_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["p_xpos", "p_ypos", "p_xvel", "p_yvel", "mal1_xpos", "mal1_ypos", "mal1_xvel", "mal1_yvel", "mal2_xpos", "mal2_ypos", "mal2_xvel", "mal2_yvel", "a_x", "a_y"])
    while not end_script:
        # Retrieve mouse position and calculate target pos on table
        mouse_pos = pyautogui.position()
        target_pos = transform(mouse_pos)

        # Update PID controllers' new setpoint (target)
        pid_x.setpoint = target_pos[0]
        pid_y.setpoint = target_pos[1]

        # Index 4 is mal1_xpos, Index 5 is mal1_ypos
        action = [pid_x(obs[4]), pid_y(obs[5])]

        obs, rew, term, _, _ = env.step(action)
        data.append(np.concatenate((obs.flatten(), action), axis=None))

        if term:
            # Only record data if it has positive reward
            if rew > REW_THRESHOLD:
                writer.writerows(data)
                episode_count += 1
                print(episode_count)
            else:
                print("Bad data, reject it")

            if episode_count >= MAX_EPISODES:
                end_script = True

            data.clear()
            env.reset()

env.close()
