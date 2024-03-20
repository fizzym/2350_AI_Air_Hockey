from keras.models import load_model
from air_hockey_gym.envs import SingleMalletBlockEnv
import numpy as np

'''
Script to qualitatively validate the performance of imitation learning agent. Uses continuous action space
but plan to switch to discrete action space.
'''

model_x = load_model('../../trained_models/model_x.h5')
model_y = load_model("../../trained_models/model_y.h5")

SIMULATION_STEPS = 1000
MAX_ACCEL = 5

max_rew = 5.0

# Adjusted the puck and opponent mallet speed to give easier shots 
env = SingleMalletBlockEnv(max_reward=max_rew, render_mode="human", puck_box=[(0.10,0.35), (0.10,-0.35)], mal2_vel_range = [0.5,0.5])

obs, _ = env.reset()

for i in range(SIMULATION_STEPS):
    new_obs = obs.reshape(12)
    action = [model_x.predict(new_obs.reshape(1, -1)), model_y.predict(new_obs.reshape(1, -1))]
    action = np.squeeze(action)

    obs, rew, term, _, _ = env.step(action)
    
    if term:
        env.reset()

env.close()
