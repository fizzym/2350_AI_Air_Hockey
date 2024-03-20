from keras.models import load_model
from air_hockey_gym.envs import SingleMalletShootEnvV2
import numpy as np

'''
Script to qualitatively validate the performance of imitation learning agent. Uses discrete action space
to be consistent with current training and for easier validation. 
'''

model = load_model('../../trained_models/im_learning_discrete_model_v2.h5')

SIMULATION_STEPS = 10000
MAX_ACCEL = 5

max_rew = 5.0

# Adjusted the puck and opponent mallet speed to give easier shots 
env = SingleMalletShootEnvV2(max_reward=max_rew, render_mode="human", puck_box=[(-0.95,0.45),(-0.05,-0.45)],
                              mal1_box= [(-0.95,0.45),(-0.05,-0.45)], discrete_actions=True)

obs, _ = env.reset()

for i in range(SIMULATION_STEPS):
    action = model.predict(obs.reshape(1,-1))
    a = np.argmax(action)

    print(a)

    obs, rew, term, _, _ = env.step(a)
    
    if term:
        env.reset()

env.close()
