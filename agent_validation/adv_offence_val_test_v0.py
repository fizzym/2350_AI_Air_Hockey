from agent_validation.val_test import ValidationTest
from agents.rl_agent import RL_Agent
from air_hockey_gym.envs.single_mallet_shooting_v2 import SingleMalletShootEnvV2
from torch.utils.tensorboard import SummaryWriter 
import pickle 
import numpy as np


class AdvOffenceValTest(ValidationTest):
    
    def __init__(self, **kwargs):
        """Perform any required initialization.
        """
        pass

    def test_agent(self, agent : RL_Agent, log_path : str, render_mode, data_name : str,
                   discrete_actions = True, accel_mag=1.0, **kwargs):
        """Perform desired test. Saves relevant statistics using tensorboard to log_path.

        Args:
            agent: The agent to test.
            log_path: Directory to save test logs to.
            render_mode: How to render environment used during testing. Options are 'human', 'rgb_array', 'depth_array' 
            data_name: Name of the recorded testing data to use. Should be in agent_validation/data directory
            accel_mag: The magnitude of acceleration of the agent mallet
        """

        env = SingleMalletShootEnvV2(render_mode = render_mode, discrete_actions = discrete_actions)
        tb = SummaryWriter(log_path)

        testing_data = []

        with open("agent_validation/data/" + data_name, "rb") as file:
            testing_data = pickle.load(file)

        num_attempts = len(testing_data)
        num_goals = 0
        for k in range(num_attempts):

            #Grab data for current episode and reset environment to recorded ICs
            obs = env.set_custom_state(testing_data[k], np.zeros((6,1)))

            terminated = False

            #For each episode run through recorded actions + buffer
            while not terminated:
                action = agent.predict(obs)

                obs, rew, terminated, _, info_dict = env.step(action)

                if terminated:
                    if info_dict["termination_reason"] == "Goal scored on opponent":
                        num_goals += 1

                    #Reset model so step counter is also reset
                    env.reset()
            
            
        missed_shots = num_attempts - num_goals
        tb.add_scalar("Advanced Offence Test - Missed Shots (" + str(num_attempts) + " Attempts)", missed_shots, 0)
        print("Agent missed", missed_shots, "shots out of", num_attempts, "attempts")          

        tb.close()
