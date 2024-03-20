from air_hockey_gym.envs import SingleMalletShootEnvV2

import time
import numpy as np
import pyautogui
from simple_pid import PID
import csv
import datetime
from pynput.keyboard import Listener, KeyCode
from stable_baselines3 import PPO

'''
### Data Collection File For Agent Offensive Training
This file is used to collect observations and action data for the purpose of Imitation Learning.
Changes: This script allows the user to play against a trained agent to collect more advanced playing data.

When the simulation begins, the mouse cursor is used to control the player mallet. The goal is to play as well as possible against the
opponent and then save the observations and actions as a row vector. 

The user determines if the data should be saved or discarded. The user MUST manually save the data, ideally at regular intervals. The script stores
a running list of rows. Pressing the discard key will clear the currently stored data. Pressing the save key will write the currently stored data to
the output CSV file, and then clear the data. Both keys will clear the running data.
Key commands: F key to discard data, G key to save data.

The collected data can be used with Cross Entropy Agents, since it uses episodes to complete its training. The goal is to "seed" the
agent with high quality data so that it can learn desirable behaviours early.
'''

# Set data collection parameters
TESTING_DELAY = 0  # seconds
MAX_MALLET_ACCEL = 5  # mallet acceleration
REW_THRESHOLD = 0  # Episode reward threshold
SCREEN_X, SCREEN_Y = pyautogui.size()

# Environment parameters
DISCRETE_ACTIONS = True
# Distance camera is above the table. Recommended to adjust so that table fills majority of screen
# so that PID control is more intuitive
CAMERA_DIST = 1.5

#The approximate dimensions of the displayed environment (playing surface is 2mx1m
#but more of the table will be displayed based on camera height and screen dimensions)
#Used to determine transformation from mouse position to table position 
DISP_LENGTH = 2.0
DISP_WIDTH = 1.25

# Special camera angle for data collection
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": CAMERA_DIST,
    "lookat": np.array((0, 0, 0)),
    "elevation": -90,
}

# Define reset and save data key
# Reset key resets all positions but discards data collected
# Save key resets all positions and saves data to csv file
RESET_KEY = KeyCode(char='f')
SAVE_KEY = KeyCode(char='g')

# Variable to track when reset or save data key is pressed
reset_triggered = False
save_triggered = False

# Define the function that will be called when a key is pressed
def on_press(key):
    global reset_triggered
    global save_triggered
    if key == RESET_KEY:
        reset_triggered = True
    elif key == SAVE_KEY:
        save_triggered = True

# Start the listener
listener = Listener(on_press=on_press)
listener.start()

# Formatting file name and path
cur_datetime = datetime.datetime.now()
formatted_datetime = cur_datetime.strftime("%m-%d-%H-%M")

data_path = "data/"
data_file = "train_data_v2_" + formatted_datetime + ".csv"

# Transform a coordinate on screen to an Air Hockey table coordinate
def transform(point):
    new_x = min((DISP_LENGTH * point[0] / SCREEN_X) - DISP_LENGTH/2.0, 0)
    new_y = -(DISP_WIDTH* (point[1] / SCREEN_Y) - DISP_WIDTH/2.0)
    return new_x, new_y

PUCK_BOX = [(-0.95,0.45),(-0.05,-0.45)]
MAL1_BOX = [(-0.95,0.45),(-0.05,-0.45)]
MAL2_BOX = [(0.05,-0.45),(0.95,0.45)]

# Randomize the mallet and puck spawn positions across their entire playable area
env = SingleMalletShootEnvV2(use_both_agents=True,
    discrete_actions=DISCRETE_ACTIONS, render_mode="human", default_camera_config=DEFAULT_CAMERA_CONFIG)

# Load best performing PPO agent as the opponent
model = PPO.load("../../trained_models/ppo_alt3")

pid_x = PID(15, 1, 2, setpoint=0)
pid_y = PID(15, 1, 2, setpoint=0)

time.sleep(TESTING_DELAY)

data = []

# Initialize mallets and puck position
obs = env.spawn_in_box(MAL1_BOX, PUCK_BOX, MAL2_BOX)

with open(data_path + data_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["p_xpos", "p_ypos", "p_xvel", "p_yvel", "mal1_xpos", "mal1_ypos", "mal1_xvel", "mal1_yvel", "mal2_xpos", "mal2_ypos", "mal2_xvel", "mal2_yvel", "a_x", "a_y"])
    
    while True:
        env.render()
        a, _ = model.predict(obs["mal2"])
        #Agent is RHS mallet in this environment, so need to negate actions so they are in world coord system
        
        agent_action = np.multiply(-1, list(env.actions[a])) if DISCRETE_ACTIONS else np.multiply(-1, a)

        # Retrieve mouse position and calculate target pos on table
        mouse_pos = pyautogui.position()
        target_pos = transform(mouse_pos)

        # Update PID controllers' new setpoint (target)
        pid_x.setpoint = target_pos[0]
        pid_y.setpoint = target_pos[1]

        # Index 4 is mal1_xpos, Index 5 is mal1_ypos
        obs_1 = obs["mal1"]
        human_action = [pid_x(obs_1[4]), pid_y(obs_1[5])]

        #Manually step sim and check if goal is scored since base class env does not handle this
        obs = env.step_sim(human_action, agent_action)
        goal_scored, _ = env._check_goal_scored()
        data.append(np.concatenate((obs_1.flatten(), human_action), axis=None))


        #If goal is scored or user hits reset button, reset environment 
        if goal_scored:
            obs = env.spawn_in_box(MAL1_BOX, PUCK_BOX, MAL2_BOX)
        # If reset triggered, discards data and resets default positions
        if reset_triggered:
            print("Resetting and discarding data")
            data.clear()
            obs = env.spawn_in_box(MAL1_BOX, PUCK_BOX, MAL2_BOX)
            reset_triggered = False
        # If save triggered, save data and resets default positions
        elif save_triggered:
            print("Resetting and saving data")
            writer.writerows(data)
            data.clear()
            obs = env.spawn_in_box(MAL1_BOX, PUCK_BOX, MAL2_BOX)
            save_triggered = False

env.close()
