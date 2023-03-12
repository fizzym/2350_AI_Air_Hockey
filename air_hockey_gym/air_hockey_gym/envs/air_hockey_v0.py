from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
import os
import mujoco
import time

import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.4,
}

#TODO see if I need to add EzPickle
class AirHockeyEnv(MujocoEnv):
    """

    ### Mallet Definition
	Mallet 1 - Mallet that plays in -x direction from table coordinate frame.
	The mallet on the left side of the table when looking from default camera angle. 
	

	Mallet 2 - Mallet that plays in +x direction from table coordinate frame.
	The mallet on the right side of the table when looking from default camera angle.

	### Observation 
	Dictionary of mallet name to (3,4) numpy array containing positions and velocities of all objects (puck, mallet 1, mallet 2)

	Keys:
	mal_1 - Mallet 1
	mal_2 - Mallet 2

	Values:
	(3,4) np array (obs)
	Each row of the array represents [x position, y position, x velocity, y velocity] for a given object.

	Coordinates are in reference frame of controlled mallet (mallet that corresponds to dict key used). 
	Mallet reference frame is defined with origin at centre of table, x-y plane in the plane of the table,
	and the positive x-axis pointing away from the controlled mallet's goal. This is done so observation
	is independent of which side of the table the mallet is on. 

	Row 0 = Puck
	Row 1 = Controlled Mallet
	Row 2 = Opponent Mallet

	Ex. obs[0, 2] = x position of the puck in the controlled mallet's reference frame


	### Action
	(4,) numpy array (act) containing accelerations of both mallet's

	act[:2] = [x1 acceleration, y1 accleration] acceleration of Mallet 1 in Mallet 1 reference frame
	act[2:] = [x2 acceleration, y2 accleration] acceleration of Mallet 2 in Mallet 2 reference frame  

	### Reward
	Dict of mallet name to reward (int)

	Keys:
	mal_1 - Mallet 1 
	mal_2 - Mallet 2 

	Value:
	#TODO describe reward definition


	### Episode End
	A training episode terminates when the centre point of the puck passes the goal line in either goal


	

	"""

    metadata = {
        #All 3 render modes required for MujocoEnv
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ],

        #Should be equal to frame_skip / timestep. timestep defined in xml. 
        #frame_skip defined in __init__
        "render_fps": 25,
    }


    def __init__(self, max_reward=10, **kwargs):

    	#Name of agents used as keys in dictionaries
        self.mal1_name = "mal1"
        self.mal2_name = "mal2"

        #Store max reward
        self.max_reward = max_reward

        #Distance of goal line from center of table
        self.goal_dist = 1.0
        self.goal_width = 0.13

        #Note: Other MuJoCo Envs do not seem to define limits to observation space
        #The limits don't seem to be checked/enforced anywhere so I have also not included them
        box_obs = Box(low=-np.inf, high=np.inf, shape=(3,4), dtype=np.float64)
        observation_space = Dict({self.mal1_name : box_obs, self.mal2_name : box_obs})
        
        self.asset_path = os.path.join(os.path.dirname(__file__), "assets/")
        print(self.asset_path)

        MujocoEnv.__init__(
            self,
            self.asset_path + "table_2_mallets.xml",
            #Defines how many time steps should be executed between each step function call 
            frame_skip=40,
            observation_space=observation_space,
            #TODO figure out how to define camera config
            #default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    #@param a - action to be undertaken - shape (4,)
    #See documentation above
    def step(self, a):
        
        #Copy action so that we do not change original list 
        a_copy = list(a)
      
        #Acceleration of mallet 2 is in it's reference frame, which is rotated 180 degrees
        #from the reference frame of the actuators. Rotate back to input to MuJoCo
        a_copy[2] *= -1
        a_copy[3] *= -1

        self.do_simulation(a_copy, self.frame_skip)
        ob = self._get_obs()

        reward = {self.mal1_name : self._get_mal_1_rew(ob[self.mal1_name]), 
        self.mal2_name : self._get_mal_2_rew(ob[self.mal2_name]) }

        #Check if centre of puck is in goal
        #Puck x position is first defined joint, puck y position is 2nd defined joint 
        #and (0,0) at center of table
        #Goal line is 1m in x from center and goal is 26 cm wide
        terminated = np.abs(self.data.qpos[0]) > self.goal_dist and np.abs(self.data.qpos[1]) < self.goal_width

        #TODO determine if truncated condition is needed
        truncated = False

        #TODO define debug info or other useful metrics in info
        info = {}

        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, truncated, info

    
    def reset_model(self):
        
        #Reset all values to default from XML
        qpos = np.zeros(13,)
        qvel = np.zeros(13,)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):

        #Calculate current Cartesian coordinates
        mujoco.mj_kinematics(self.model,self.data)

        #Note: All frames have same orientation, but different origins. Therefore qvel values
        #are still in cartesian coords

        #Puck's DOF are defined 1st and 2nd in the XML
        cart_p = np.concatenate((self.data.geom("puck").xpos[:2], self.data.qvel[0:2]))
    	#Mallet 1's DOF are defined 4th and 5th in the XML
        cart_m1 = np.concatenate((self.data.geom("mallet1").xpos[:2], self.data.qvel[3:5]))
        #Mallet 2's DOF are defined 7th and 8th in the XML
        cart_m2 = np.concatenate((self.data.geom("mallet2").xpos[:2], self.data.qvel[6:8]))
        

        m1_obs = np.stack([cart_p, cart_m1, cart_m2])
        
        #Mallet 2 reference frame is 180 rotation of Mallet 1 frame, therefore multiply all values by -1
        m2_obs = -1 * np.stack([cart_p, cart_m2, cart_m1])

        return {self.mal1_name: m1_obs, self.mal2_name : m2_obs}

 
 	#@param - obs_1 Current observation of the table environment from mallet 1's perspective
 	#@returns double Reward for mallet 1 based on the current environment state
    def _get_mal_1_rew(self, obs_1):
    	return self._get_rew_from_obs(obs_1)


    #@param obs_2 -  Current observation of the table environment from mallet 2's perspective
 	#@returns double - Reward for mallet 2 based on the current environment state
    def _get_mal_2_rew(self, obs_2):
    	return self._get_rew_from_obs(obs_2)

    #Helper function to calculate portions of reward which only depend on current observation
    #@param obs - Observation from one mallet's perspective
    def _get_rew_from_obs(self, obs):
    
        rew = 0

        #If scored on opponent, return max reward
        if obs[0,0] > self.goal_dist and np.abs(obs[0,1]) < self.goal_width:
            rew = self.max_reward

        #If opponent scored on you, return negative max reward    
        elif obs[0,0] < -self.goal_dist and np.abs(obs[0,1]) < self.goal_width:
            rew = -self.max_reward

        #If goal not scored, do typical reward calculation    
        else:
            #If puck on opponent's side, add reward
            if(obs[0,0] > 0):
                rew += 0.2 * self.max_reward

            #If puck on our side, subtract reward
            elif(obs[0,0] < 0):
                rew -= 0.2 * self.max_reward

        return rew



