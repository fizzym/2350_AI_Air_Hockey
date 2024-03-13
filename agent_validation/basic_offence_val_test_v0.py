from agent_validation.val_test import ValidationTest
from agents.rl_agent import RL_Agent
from air_hockey_gym.envs.single_mallet_shooting_v2 import SingleMalletShootEnvV2
from torch.utils.tensorboard import SummaryWriter 

#Define map which translates IC used to text for printout
ic_map = {0: "straight ahead", 1 : "to left (from agent's perspective)", 2: "to right (from agent's perspective)"}

#Number of episodes to run for each IC
NUM_EPISODES = 100

class BasicOffenceValTest(ValidationTest):
    
    def __init__(self, **kwargs):
        """Perform any required initialization.
        """
    
        #Define initial conditions for test. First is straight on shot, second and third are angled shots from opposite edges of table
        self.pos_ICs = [[-0.5, 0, -0.8, 0, 0.9, 0.4], [-0.5, 0.25, -0.8, 0, 0.9, 0.4], [-0.5, -0.25, -0.8, 0, 0.9, 0.4]]
        self.vel_ICs =  [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]


    def test_agent(self, agent : RL_Agent, log_path : str, render_mode, discrete_actions = True, **kwargs):
        """Perform desired test. Saves relevant statistics using tensorboard to log_path.

        Args:
            agent: The agent to test.
            log_path: Directory to save test logs to.
            render_mode: How to render environment used during testing. Options are 'human', 'rgb_array', 'depth_array' 

        """

        env = SingleMalletShootEnvV2(render_mode = render_mode, discrete_actions = discrete_actions)
        tb = SummaryWriter(log_path)
        
        for k in range(len(self.pos_ICs)):
            pos = self.pos_ICs[k]
            vel = self.vel_ICs[k]
            num_goals = 0

            for i in range(NUM_EPISODES):
                obs = env.set_custom_state(pos, vel)

                terminated = False
                while not terminated:
                    action = agent.predict(obs)

                    obs, rew, terminated, _, info_dict = env.step(action)

                    if terminated:
                        if info_dict["termination_reason"] == "Goal scored on opponent":
                            num_goals += 1
                        
                        #Reset model so step counter is also reset
                        env.reset()
            
            missed_shots = NUM_EPISODES - num_goals

            tb.add_scalar("Basic Offensive Test - Missed Shots (" + str(NUM_EPISODES) + " Attempts)", missed_shots, k)
            print("Agent missed", missed_shots, "shots out of", NUM_EPISODES, "attempts from puck", ic_map[k])          

        tb.close()
