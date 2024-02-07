from air_hockey_gym.envs import SingleMalletShootEnvV2
import time
import numpy as np
import pyautogui
from simple_pid import PID
import csv
import pickle
import datetime

'''
### Data Collection File
This file is used to collect initial conditions and action data for the purpose of defensive testing.

When the simulation begins, the mouse cursor is used to control the player mallet. The puck will be spawned randomly
and the purpose is to record data of goals being scored in a human manner. This recorded data can then be used to
test agents with more realistic shots. 

Data is only recorded if a goal is scored and data is only saved to file at the end of the script (i.e. after
desired number of goals is scored). Data saved to agent_validation/data

The collected data is designed to be used with adv_defence_val_test_v0.py. It is saved as a pickle and has the following form
list[dict{str:numpy array}], where each list index represents data saved from a different episode and the data is stored
in a dictionary with the following key/values:
{
    "ics" : (6,1) numpy array of initial positions of all objects
    "actions" : list[(2,1) numpy arrays] where the array at index j represents the action taken at timestep j
                (length of list will vary based on episode length) 
}
'''

# Set data collection parameters
NUM_GOALS = 50 #Number of goals to record data for
TESTING_DELAY = 3  # Delay before starting to run test in seconds
MAX_MALLET_ACCEL = 5  # mallet acceleration
# Distance camera is above the table. Recommended to adjust so that table fills majority of screen
# so that PID control is more intuitive
CAMERA_DIST = 1.4 
SCREEN_X, SCREEN_Y = pyautogui.size()

# Special camera angle for data collection
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": CAMERA_DIST,
    "lookat": np.array((0, 0, 0.1)),
    "elevation": -90,
}

# Formatting file name and path
cur_datetime = datetime.datetime.now()
formatted_datetime = cur_datetime.strftime("%m-%d-%H-%M")

data_path = "agent_validation/data/"
data_file = "adv_def_test_data_" + str(NUM_GOALS) + "_eps_" + formatted_datetime + ".pkl"

# Transform a coordinate on screen to an Air Hockey table coordinate
def transform(point):
    new_x = min((2 * point[0] / SCREEN_X) - 1, 0)
    new_y = -((point[1] / SCREEN_Y) - 0.5)
    return new_x, new_y

#Helper function to pick out object positions from observation
def get_pos_from_obs(obs):
    return np.concatenate((obs[0:2],obs[4:6], obs[8:10]))

max_rew = 1.0
# Randomize the  controlled mallet and puck spawn positions across their entire playable area
# Place opponent mallet in corner so it does not interfere
env = SingleMalletShootEnvV2(max_reward=max_rew, puck_box=[(-0.95,0.45),(-0.05,-0.45)],
                             mal1_box= [(-0.95,0.45),(-0.05,-0.45)],
                             mal2_box = [(0.9,0.45), (0.9,.45)],
                             discrete_actions=False, render_mode="human", default_camera_config=DEFAULT_CAMERA_CONFIG)

pid_x = PID(15, 1, 2, setpoint=0)
pid_y = PID(15, 1, 2, setpoint=0)

env.render()
time.sleep(TESTING_DELAY)

obs, _ = env.reset()

#Initialize data storage objects
num_goals = 0
data = []
ics = get_pos_from_obs(obs)
action_list = []

while num_goals < NUM_GOALS:
    
    # Retrieve mouse position and calculate target pos on table
    mouse_pos = pyautogui.position()
    target_pos = transform(mouse_pos)

    # Update PID controllers' new setpoint (target)
    pid_x.setpoint = target_pos[0]
    pid_y.setpoint = target_pos[1]

    # Index 4 is mal1_xpos, Index 5 is mal1_ypos
    action = [pid_x(obs[4]), pid_y(obs[5])]
    action_list.append(action)

    obs, rew, term, _, info_dict = env.step(action)

    if term:
        # Only record data if goal was scored
        if info_dict["termination_reason"] == "Goal scored on opponent":
            
            storage_dict = {}
            storage_dict["ics"] = ics
            storage_dict["actions"] = action_list.copy()
            data.append(storage_dict)
            
            num_goals += 1
            print("Goals scored:", num_goals)
        else:
            print("No goal scored, not saving data")

        #Store controlled mallet current position, then manually set it after reset
        #so mallet does not jump around
        mal1_pos = obs[4:6]
        obs, _ = env.reset()
        obs = env.set_custom_state(np.concatenate((obs[0:2], mal1_pos, obs[8:10])), np.zeros((6,)))
        ics = get_pos_from_obs(obs) #Store new ICs for saving if goal is scored
        action_list.clear() 
        env.render()

with open(data_path + data_file, 'wb') as file:
    pickle.dump(data, file)
            

env.close()
