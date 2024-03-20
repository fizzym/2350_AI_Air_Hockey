from agent_validation.val_test import ValidationTest
from agents.rl_agent import RL_Agent
from air_hockey_gym.envs.air_hockey_base_class_v0 import AirHockeyBaseClass
from torch.utils.tensorboard import SummaryWriter 
import pickle 
import numpy as np

#Number of steps to continue each simulation after recorded actions are completed
STEP_BUFFER = 50

class AdvDefenceValTest(ValidationTest):
    
    def __init__(self, **kwargs):
        """Perform any required initialization.
        """
        pass

    def test_agent(self, agent : RL_Agent, log_path : str, render_mode, data_name : str,
                   discrete_actions = True, accel_mag = 1.0, **kwargs):
        """Perform desired test. Saves relevant statistics using tensorboard to log_path.

        Args:
            agent: The agent to test.
            log_path: Directory to save test logs to.
            render_mode: How to render environment used during testing. Options are 'human', 'rgb_array', 'depth_array' 
            data_name: Name of the recorded testing data to use. Should be in agent_validation/data directory
            accel_mag: The magnitude of acceleration of the agent mallet
        """

        env = AirHockeyBaseClass(render_mode = render_mode, use_both_agents = True, 
        discrete_actions = discrete_actions, accel_mag=accel_mag)
        tb = SummaryWriter(log_path)

        testing_data = []

        with open("agent_validation/data/" + data_name, "rb") as file:
            testing_data = pickle.load(file)

        num_goals = 0
        num_eps = len(testing_data)
        for k in range(num_eps):

            #Grab data for current episode and reset environment to recorded ICs
            cur_data = testing_data[k]
            #Modify agent IC (RHS mallet) so that it spawns in middle of table
            ics = cur_data["ics"]
            ics[4:6] = [0.8, 0]
            obs = env.set_custom_state(ics, np.zeros((6,1)))
            
            if render_mode == "human":
                env.render()

            terminated = False
            step = 0
            rec_acts = cur_data["actions"]
            num_rec_acts = len(rec_acts)

            #For each episode run through recorded actions + buffer
            while step < num_rec_acts + STEP_BUFFER:
                if render_mode == "human":
                    env.render()

                a = agent.predict(obs["mal2"])

                #Agent is RHS mallet in this test, so need to negate actions so they are in world coord system
                action = np.multiply(-1, list(env.actions[a]))

                #Use recorded action if one exists otherwise no action
                opp_action = [0,0] if step >= num_rec_acts else rec_acts[step]

                #Manually step sim and check if goal is scored since base class env does not handle this
                obs = env.step_sim(opp_action, action)
                step +=1
                goal_scored, net_scored = env._check_goal_scored()
                
                if goal_scored:
                    if net_scored == 1:
                        num_goals += 1
                    
                    break
            
            
        tb_string = "Advanced Defence Test - Goals Scored On Agent (" + str(num_eps) + " Attempts)"
        tb.add_scalar(tb_string, num_goals, 0)
        print("Agent was scored on", num_goals, "times out of", len(testing_data), "attempts")          

        tb.close()
