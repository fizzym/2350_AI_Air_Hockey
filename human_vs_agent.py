import pyautogui
import numpy as np
from simple_pid import PID
from pynput.keyboard import Listener, KeyCode

from air_hockey_gym.envs.air_hockey_base_class_v0 import AirHockeyBaseClass
from end_to_end_utils import get_agent_class_and_config

# ****************************************
# Script which allows a human to play against agent.
#
# Must be run using `sudo python3 humand_vs_agent.py` to allow for resettting environment via key press. 
#
# Loads environment with human as LHS mallet and agent as RHS mallet. Steps through environment
# indefinitely using mouse tracking to determine human mallet actions. Resets when goal is scored
# or f key is pressed.
#
# Change which agent to use and some paramters using below variables

AGENT_TYPE = "ppo_agent"
AGENT_VERSION = "v0"
#Path to saved agent to load
AGENT_PATH = "trained_models/ppo_alt_5to3acc_4/ppo_agent_v0.zip"
DISCRETE_ACTIONS = True
ACCEL_MAG = 3.0 #Magnitude of acceleration of agent (if discrete)
AGENT_START_X = 0.4
# Distance camera is above the table. Recommended to adjust so that table fills majority of screen
# so that PID control is more intuitive
CAMERA_DIST = 1.5 

#The approximate dimensions of the displayed environment (playing surface is 2mx1m
#but more of the table will be displayed based on camera height and screen dimensions)
#Used to determine transformation from mouse position to table position 
DISP_LENGTH = 2.5
DISP_WIDTH = 1.25

SCREEN_X, SCREEN_Y = pyautogui.size()

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": CAMERA_DIST,
    "lookat": np.array((0, 0, 0.0)),
    "elevation": -90,
}

# Define reset key
TARGET_KEY = KeyCode(char='f')

# Variable to track when reset key is pressed
reset_triggered = False

# Define the function that will be called when a key is pressed
def on_press(key):
    global reset_triggered
    if key == TARGET_KEY:
        reset_triggered = True

# Start the listener
listener = Listener(on_press=on_press)
listener.start()

def transform(point):
    new_x = min((DISP_LENGTH * point[0] / SCREEN_X) - DISP_LENGTH/2.0, 0)
    new_y = -(DISP_WIDTH* (point[1] / SCREEN_Y) - DISP_WIDTH/2.0)
    return new_x, new_y


env = AirHockeyBaseClass(use_both_agents=True, discrete_actions=DISCRETE_ACTIONS,
                         render_mode="human", default_camera_config=DEFAULT_CAMERA_CONFIG,
                         accel_mag=ACCEL_MAG)

#Pack variables into dictionary to use utility function
agent_info = {"agent_name" : AGENT_TYPE,
              "version" : AGENT_VERSION}
#Get agent class and config info
agent_class, agent_config = get_agent_class_and_config(agent_info)

#Create agent class with specified params
agent = agent_class(filepath=AGENT_PATH, **agent_config["init_params"])

pid_x = PID(15, 1, 2, setpoint=0)
pid_y = PID(15, 1, 2, setpoint=0)

#Initialize ICs so mallets spawn at same distance from middle
pos_ics = np.zeros((6,1))
pos_ics[2] = -AGENT_START_X
pos_ics[4] = AGENT_START_X
obs = env.set_custom_state(pos_ics, np.zeros((6,1)))

while True:
    env.render()
    a = agent.predict(obs["mal2"])
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

    #If goal is scored or user hits reset button, reset environment 
    if goal_scored or reset_triggered:
        #Reset puck to center, agent to its start position, but keep human mallet
        #in same position so it does not jump around on user
        pos_ics = np.concatenate(([0,0], obs["mal1"][4:6], [AGENT_START_X,0]))
        obs = env.set_custom_state(pos_ics, np.zeros((6,1)))
        reset_triggered = False
