from air_hockey_gym.envs.air_hockey_base_class_v0 import AirHockeyBaseClass   
#from air_hockey_base_class_v0 import AirHockeyBaseClass
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Discrete
import os
import mujoco

import numpy as np
from numpy.random import uniform as unif

PUCK_STATIONARY_THRESH = 1e-4
MAX_VEL = 3.0

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.4,
}

class SingleMalletBlockEnv(AirHockeyBaseClass):
    """
    V3 Update:
    - Refactor env to implement AirHockeyBaseClass. 
    - Added option to specify discrete or continous action space.

    ###Environment Description

    Modified version of air hockey environment used for training single agent to block puck shots
    Randomly spawns puck on opponent's side, opponent mallet behind puck with some initial velocity toward
    the controlled goal. Controlled mallet spawns randomly in front of it's own goal.
    Majority of spawning variables are customizable (see below init function documention)

	### Observation 
	(12,1) numpy array containing positions and velocities of all objects (puck, mallet 1, mallet 2)

	Values:
	(12,1) np array (obs)
	Every 4 values represents [x position, y position, x velocity, y velocity] for a given object. 
    Order is puck, controlled mallet, opponent mallet. E.g. obs(4:8) are the values for the controlled mallet

	Coordinates are in reference frame of controlled mallet
	Mallet reference frame is defined with origin at centre of table, x-y plane in the plane of the table,
	and the positive x-axis pointing away from the controlled mallet's goal.


	### Action
	If environment is continous, should be (2,) numpy array of accelerations.
    If environment is discrete, should be integer representing which direction to move in. 

	### Reward
	
	Value:
	All rewards are given in terms of percent of max_reward argument (default 10)
    "=" indicates reward takes that value and is not changed based on any other conditions
    "+=" indicates if condition is met that amount is added to current reward, in addition to any other conditions met

    Opponent scored on you = -100%
    Scored on opponent = 100%
    Hit puck towards opponent += 10%
    Hit anything other than puck -= 50%
    Puck crosses back to opponent side += 10% * puck x velocity

	### Episode End
	A training episode terminates when the any of the following conditions are met:
    - A goal is scored
    - 200 steps have past (reasonable value for which puck movement mostly stops)

	"""

    metadata = {
        #All 3 render modes required for MujocoEnv
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ],

        #Should be equal to  1/(frame_skip * timestep). timestep defined in xml. 
        #frame_skip defined in __init__
        "render_fps": 25,
    }

    
    def __init__(self, max_reward=1, puck_box=[(0.10,0.25), (0.55,-0.25)], mal2_puck_dist_range = [0.2,0.3],
                 mal2_vel_range = [0.5,2], mal1_box= [(-.90,0.15),(-0.7,-0.15)],
                 max_accel=5, discrete_actions = True,  **kwargs):
                 """
                 All coordinates are in world coordinate frame (center of table, 
                 positive x towards opponent goal)

                 Inputs:
                    max_reward: Maximum reward to return. All rewards are given as percent of this value.
                    puck_box: Bounding box which puck will spawn uniformly in.
                            [(x1,y1),(x2,y2)] Where (x1,y1) are coordinates of top left corner of box, 
                            and (x2,y2) is bottom right
                    mal2_puck_dist_range: Absolute value of range of distances between mallet 2 and the puck
                    mal2_vel: Absolute value of range of velocites mallet 2 can spawn with
                    mal1_box: Bounding box which mallet1 will spawn uniformly in. Same format as puck_box.
                    max_accel: The max acceleration of the agent mallet
                    discrete_actions: Flag to set if environment's action space should be discrete or continous.
                              Will be continous if False, discrete if True
        
                 """
                 super().__init__(max_reward=max_reward, max_accel = max_accel,
                                  discrete_actions= discrete_actions, **kwargs)
       

                 #Store parameters for use in reset function
                 self.p_box = puck_box
                 self.m2_puck_dist = mal2_puck_dist_range
                 self.m2_vel = mal2_vel_range
                 self.m1_box = mal1_box
                
                 self.discrete_act = discrete_actions
                 #Initialize step counter
                 self.num_steps = 0


    def step(self, a):
        """Steps the simulation forward by 1 timestep.

        Inputs:
            a: Action to use. Should be integer for discrete, length 2 array for continous.

        Outputs: (obs, reward, termination, truncated, info)
            obs: Observation of state (see class description).
            reward: Reward for previous step
            termination: bool indicating if environment is finished (True if finished)
            truncated: bool indicating if environment was truncated
            info: Dictionary with environment info
        """

        if self.discrete_act:
            action = list(self.actions[a])
        else:
            action = a

        self.step_sim(action)
        ob = self._get_obs()

        self.num_steps += 1

        reward = self._get_rew(ob)

        info = {}
    
        #Check if goal scored
        goal_scored, net_scored = self._check_goal_scored()
    
        #Set termination
        terminated = goal_scored or self.num_steps > 200

        #TODO determine if truncated condition is needed
        truncated = False

        #Add rationale for ending episode
        if terminated:
            term_reason = "Goal scored" if goal_scored else "Max steps reached"

            if goal_scored:
                term_reason += " on opponent" if net_scored == 1 else " on agent"

            info["termination_reason"] = term_reason
            
        #TODO determine if truncated condition is needed
        truncated = False

        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, truncated, info

    
    def reset_model(self):

        self.num_steps = 0

        #Spawn puck and agent within desired box
        temp_obs = self.spawn_in_box(self.m1_box, self.p_box, [(0,0), (0,0)])
        
        #Pick out coordinates
        puck_x = temp_obs[0]
        puck_y = temp_obs[1]

        m1_x = temp_obs[4]
        m1_y = temp_obs[5]

        #Pick mallet 2 start location behind puck
        m2_dist = unif(self.m2_puck_dist[0], self.m2_puck_dist[1])
        theta = np.arctan(puck_y/(puck_x + 1))
        m2_x = puck_x + np.cos(theta) * m2_dist
        m2_y = puck_y + np.sin(theta) * m2_dist

        #Pick mallet 2 initial velocity
        m2_vel = - unif(self.m2_vel[0], self.m2_vel[1])
        m2_vel_x = m2_vel * np.cos(theta)
        m2_vel_y = m2_vel * np.sin(theta)

        pos = [puck_x, puck_y, m1_x, m1_y, m2_x, m2_y]
        vel = [0,0,0,0,m2_vel_x,m2_vel_y]

        return self.set_custom_state(pos,vel)
    

 
 	#@param - obs Current observation of the table environment
 	#@returns double Reward for on the current environment state
    def _get_rew(self, obs):
        """ Get reward given an observation of the state.

        Inputs:
            obs: Observation of the environment state.

        Outputs:
            reward (float): Reward for current state.
        """

        
        #If opponent scored on you, return negative max reward
        goal_scored, net_scored = self._check_goal_scored()    
        if net_scored == 2:
            return -self.max_reward
        elif net_scored == 1:
            return self.max_reward

        rew = 0 
        #Iterate through collisions and give positive reward if mallet hits puck
        #Give negative reward for colliding with anything else
        for i in range(self.data.ncon):

            obj_1 = self.model.geom(self.data.contact[i].geom1).name
            obj_2 = self.model.geom(self.data.contact[i].geom2).name
            coll_set = {obj_1, obj_2}

            if "mallet1" in coll_set and "puck" not in coll_set:
                rew += -0.15 * self.max_reward

        #Reward based on x pos of puck
        rew += 0.001 * self.data.qpos[0] / self.goal_dist * self.max_reward

        #Reward based on x vel of puck when crossing middle
        if abs(self.data.qpos[0] / self.goal_dist) < 0.05 and self.data.qvel[0] > 0:
            rew += 0.5 * self.data.qvel[0] / self.goal_dist * self.max_reward

        return rew
