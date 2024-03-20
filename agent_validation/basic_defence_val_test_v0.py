from agent_validation.val_test import ValidationTest
from agents.rl_agent import RL_Agent
from air_hockey_gym.envs.single_mallet_blocking_v3 import SingleMalletBlockDiscreteEnv
from torch.utils.tensorboard import SummaryWriter 

#Define map which translates IC used to text for printout
ic_map = {0: "straight shots", 1 : "angled shots from left side (from agent's perspective)", 2: "angled shots from right side (from agent's perspective)"}
NUM_EPISODES = 100
class DefenceValTest(ValidationTest):
    
    def __init__(self, **kwargs):
        """Perform any required initialization.
        """
    
        #Define initial conditions for test. First is straight on shot, second and third are angled shots from opposite edges of table
        #See following link for derivation: https://docs.google.com/document/d/1JFZWgkHZAzNGL9JAqiXypYBr8AXa6GQ8JFpYiKQmDbI/edit#heading=h.uppldcgn3n2g  
        self.pos_ICs = [[0.5, 0, -0.8, 0, 0.65, 0], [0.5, 0.25, -0.8, 0, 0.5986, 0.26644], [0.5, -0.25, -0.8, 0, 0.5986, -0.26644]]
        self.vel_ICs =  [[0, 0, 0, 0, -1, 0], [0, 0, 0, 0, -0.9684, -0.1644], [0, 0, 0, 0, -0.9684, 0.1644]]


    def test_agent(self, agent : RL_Agent, log_path : str, render_mode, accel_mag = 1.0, **kwargs):
        """Perform desired test. Saves relevant statistics using tensorboard to log_path.

        Args:
            agent: The agent to test.
            log_path: Directory to save test logs to.
            render_mode: How to render environment used during testing. Options are 'human', 'rgb_array', 'depth_array' 
             accel_mag: The magnitude of acceleration of the agent mallet
        """

        env = SingleMalletBlockDiscreteEnv(render_mode = render_mode, accel_mag = accel_mag)
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
                        if info_dict["termination_reason"] == "Goal scored on agent":
                            num_goals += 1
                        
                        #Reset model so step counter is also reset
                        env.reset()
            
            tb.add_scalar("Basic Defence Test - Goals Scored On (" + str(NUM_EPISODES) + " Attempts)", num_goals, k)


            print("Agent was scored on", num_goals, "times out of", NUM_EPISODES,  "via", ic_map[k])          

        tb.close()
