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

"""Changes from V1
- Added set_custom_state method to set arbitrary positions and velocities for pucks and mallets
- Changed observation shape to be (12,)
- Added string to info dictionary which describes why environment was terminated
"""
class SingleMalletBlockDiscreteEnv(MujocoEnv):
    """

    ###Environment Description

    Modified version of air hockey environment used for training single agent to block puck shots
    Randomly spawns puck on opponent's side, opponent mallet behind puck with some initial velocity toward
    the controlled goal. Controlled mallet spawns randomly in front of it's own goal.
    Majority of spawning variables are customizable (see below init function documention)

    ### Mallet Definition
	Mallet 1 - The controlled mallet. Mallet that plays in -x direction from table coordinate frame.
	The mallet on the left side of the table when looking from default camera angle. 
	
	Mallet 2 - The opponent mallet. Mallet that plays in +x direction from table coordinate frame.
	The mallet on the right side of the table when looking from default camera angle.

	### Observation 
	(12,) numpy array containing positions and velocities of all objects (puck, mallet 1, mallet 2)

	Values:
	(12,) np array (obs)
	Each block of 4 values in array represents [x position, y position, x velocity, y velocity] for a given object.

    Object at given indices are:
    Puck: 0-3
    Controlled Mallet: 4-7
    Opponent Mallet: 8-11

    Ex. obs[4] = x position of the puck in the controlled mallet's reference frame

	Coordinates are in reference frame of controlled mallet
	Mallet reference frame is defined with origin at centre of table, x-y plane in the plane of the table,
	and the positive x-axis pointing away from the controlled mallet's goal. This is done so observation
	is independent of which side of the table the mallet is on. 


	### Action
	(4,) OR (2,) numpy array (act) containing accelerations of mallet 1

    act[:2] = [x1 acceleration, y1 accleration] acceleration of Mallet 1 in Mallet 1 reference frame

    Note: Only acceleration values for mallet 1 are used in this simulation. Action space is
    still (4,) because underlying XML has 4 actuators defined

    If action is (4,) act[2:] will be set to [0,0] so that mallet 2 has no applied accelerations

	### Reward
	Dict of mallet name to reward (int)

	Value:
	All rewards are given in terms of percent of max_reward argument (default 10)
    "=" indicates reward takes that value and is not changed based on any other conditions
    "+=" indicates if condition is met that amount is added to current reward, in addition to any other conditions met

    Opponent scored on you = -100%
    Hit puck += 50%
    Hit anything other than puck -= 50%
    Puck crosses back to opponent side += 50% * puck x velocity
    


	### Episode End
	A training episode terminates when the any of the following conditions are met:
    - Centre point of the puck passes the goal line 
    - Puck crosses back into opponent's side of table
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


    #TODO is there a standard way to define default values
    #Note all below distance values are in meters

    #@param puck_box Bounding box which puck will spawn uniformly in.
    #       Format:[(x1,y1),(x2,y2)] Where (x1,y1) are coordinates of top left corner of box, and (x2,y2) is bottom right
    #       In world coordinate frame (center of table, positive x towards opponent goal)

    #@param mal2_puck_dist_range Absolute value of range of distances for spawn positions of mallet 2 and the puck

    #@param mal2_vel_range Absolute value of range of velocites mallet 2 can spawn with

    #@param mal1_box Bounding box which mallet1 will spawn uniformly in.
    #       Format:[(x1,y1),(x2,y2)] Where (x1,y1) are coordinates of top left corner of box, and (x2,y2) is bottom right
    #       In world coordinate frame (center of table, positive x towards opponent goal)  
    
    def __init__(self, max_reward=10, puck_box=[(0.10,0.25), (0.55,-0.25)], mal2_puck_dist_range = [0.2,0.3], mal2_vel_range = [0.5,2], mal1_box= [(-.90,0.15),(-0.7,-0.15)], max_accel=5,  **kwargs):

        #Store max reward
        self.max_reward = max_reward

        #Set up actions
        self.max_accel = max_accel
        #Initialize 0 vector and 8 allowable directions
        self.actions = [[0.0,0.0],
                        [1.0,0.0],
                        [0.707,0.707],
                        [0.0,1.0],
                        [-0.707,0.707],
                        [-1.0,0.0],
                        [-0.707,-0.707],
                        [0.0,-1.0],
                        [0.707,-0.707]]
        self.action_space = Discrete(len(self.actions))

        #Check box coordinates are well formed i.e. point 1 is above and to the left of point 2
        assert puck_box[0][0] <= puck_box[1][0], "puck_box x coordinates are malformed. Right coordinate is less than left coordinate"
        assert puck_box[0][1] >= puck_box[1][1], "puck_box y coordinates are malformed. Top coordinate is less than bottom coordinate"
        assert mal1_box[0][0] <= mal1_box[1][0], "mal1_box x coordinates are malformed. Right coordinate is less than left coordinate"
        assert mal1_box[0][1] >= mal1_box[1][1], "mal1_box y coordinates are malformed. Top coordinate is less than bottom coordinate"

        #Check puck box bounds won't lead to overlapping objects
        #Leave at least 5cm between center of puck and center of table
        assert puck_box[0][0] >= 0.05, "puck_box x1 is less than minimum allowable value of 0.05"

        #Check that mallet 1 box bounds won't lead to overlapping objects
        #Values are limits of mallet 1 playing space plus/minus mallet 1 radius (5cm)
        assert mal1_box[0][0] >= -0.95, "mal1_box x1 is less than minimum allowable value of -0.95"
        assert mal1_box[0][1] <= 0.45, "mal1_box y1 is greater than maxinyn allowable value of 0.45"
        assert mal1_box[1][0] <= -0.05, "mal1_box x2 is greater than maximum allowable value of -0.05"
        assert mal1_box[1][1] >= -0.45, "mal1_box xy is less than minimum allowable value of -0.45"

        #Store parameters for use in reset function
        self.p_box = puck_box
        self.m2_puck_dist = mal2_puck_dist_range
        self.m2_vel = mal2_vel_range
        self.m1_box = mal1_box

        #Distance of goal line from center of table
        self.goal_dist = 1.0
        self.goal_width = 0.13

        self.num_steps = 0
        self.puck_bounce = False

        #Note: Other MuJoCo Envs do not seem to define limits to observation space
        #The limits don't seem to be checked/enforced anywhere so I have also not included them
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        
        self.asset_path = os.path.join(os.path.dirname(__file__), "assets/")

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

    #@param a - number from 0-8
    #See documentation above
    def step(self, a=0):

        action = list(self.actions[a])
        action.append(0)
        action.append(0)

        self.do_simulation(np.multiply(self.max_accel,action), self.frame_skip)
        ob = self._get_obs()

        self.num_steps += 1

        reward = self._get_rew(ob)

        info = {}
    
        #Check if centre of puck is in goal
        #Puck x position is first defined joint, puck y position is 2nd defined joint 
        #and (0,0) at center of table
        #Goal line is 1m in x from center and goal is 26 cm wide
        goal_scored = np.abs(self.data.qpos[0]) > self.goal_dist and np.abs(self.data.qpos[1]) < self.goal_width

        #Check if episode is finished
        terminated = goal_scored or self.num_steps > 200

        #Add rationale for ending episode
        if terminated:
            term_reason = "Goal scored" if goal_scored else "Max steps reached"

            if goal_scored:
                term_reason += " on opponent" if self.data.qpos[0] > 0 else " on agent"

            info["termination_reason"] = term_reason
            
        #TODO determine if truncated condition is needed
        truncated = False


        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, truncated, info

    
    def reset_model(self):

        self.puck_bounce = False
        self.num_steps = 0
        
        #Pick puck start location within defined box
        puck_x = unif(self.p_box[0][0], self.p_box[1][0])
        puck_y = unif(self.p_box[0][1], self.p_box[1][1])

        #TODO remove hard-coding of coordinate offset
        #Pick mallet 1 start location within defined box
        m1_x = unif(self.m1_box[0][0], self.m1_box[1][0]) + 0.25
        m1_y = unif(self.m1_box[0][1], self.m1_box[1][1])

        #TODO remove hard-coding of coordinate offset
        #Pick mallet 2 start location behind puck
        m2_dist = unif(self.m2_puck_dist[0], self.m2_puck_dist[1])
        theta = np.arctan(puck_y/(puck_x + 1))
        m2_x = puck_x + np.cos(theta) * m2_dist - 0.25
        m2_y = puck_y + np.sin(theta) * m2_dist

        #Pick mallet 2 initial velocity
        m2_vel = - unif(self.m2_vel[0], self.m2_vel[1])
        m2_vel_x = m2_vel * np.cos(theta)
        m2_vel_y = m2_vel * np.sin(theta)

        #Start with inital values for qpos and qvel
        qpos = self.init_qpos
        qvel = self.init_qvel

        #Set all desired coordinates
        qpos[0] = puck_x
        qpos[1] = puck_y

        qpos[3] = m1_x
        qpos[4] = m1_y

        qpos[6] = m2_x
        qpos[7] = m2_y

        qvel[6] = m2_vel_x
        qvel[7] = m2_vel_y

        self.set_state(qpos, qvel)
        return self._get_obs()

    
    def set_custom_state(self, pos, vel):
        """Set state of system (puck and mallet positions and velocities).

        Inputs:
            pos: Length 6 list-like which contains desired positions of objects in controlled mallet reference frame.
                 Format is [puck_x, puck_y, agent_mallet_x, agent_mallet_y, opp_mall_x, opp_mall_y]
            vel: Length 6 list-like which contains desired velocities of objects in controlled mallet reference frame.
                 Same format as pos, but for velocity components.

        Returns:
            obs: Current observation of system after desired state is set. (See class documentation for description.)

        Throws:
            AssertionError: if pos or vel have wrong length
        """

        assert len(pos) == 6 and len(vel) == 6

        #Start with inital values for qpos and qvel
        qpos = self.init_qpos
        qvel = self.init_qvel

        #Set puck values
        qpos[0] = pos[0]
        qpos[1] = pos[1]
        qvel[0] = vel[0]
        qvel[1] = vel[1]

        #Set mallet 1 values (controlled)
        qpos[3] = pos[2] + 0.25 #Add coordinate offset 
        qpos[4] = pos[3]
        qvel[3] = vel[2]
        qvel[4] = vel[3]

        #Set mallet 2 values (opponent)
        qpos[6] = pos[4] - 0.25
        qpos[7] = pos[5]
        qvel[6] = vel[4]
        qvel[7] = vel[5]

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
        
        m1_obs = np.concatenate([cart_p, cart_m1, cart_m2])

        return m1_obs

 
 	#@param - obs Current observation of the table environment
 	#@returns double Reward for on the current environment state
    def _get_rew(self, obs):

        
        #If opponent scored on you, return negative max reward    
        if obs[0] < -self.goal_dist and np.abs(obs[1]) < self.goal_width:
            return -self.max_reward
        elif obs[0] > self.goal_dist and np.abs(obs[1]) < self.goal_width:
            return self.max_reward

        rew = 0 
        #Iterate through collisions and give positive reward if mallet hits puck
        #Give negative reward for colliding with anything else
        for i in range(self.data.ncon):

            obj_1 = self.model.geom(self.data.contact[i].geom1).name
            obj_2 = self.model.geom(self.data.contact[i].geom2).name
            coll_set = {obj_1, obj_2}
                
            if "mallet1" in coll_set and "puck" in coll_set:
                if obs[0] > obs[4]:
                    rew += 0.1 * self.max_reward

            elif "mallet1" in coll_set:
                rew += -0.5 * self.max_reward

        #If puck crosses back into opponent's side, give positive reward based on how fast puck is moving
        if(self.data.qpos[0] > 0 and self.data.qvel[0] > 0):
            rew += 0.1 * self.data.qvel[0] * self.max_reward


        #Cap reward to be in range [-max_reward, max_reward]
        # if rew < 0:
        #     rew = max(rew, -self.max_reward)
        # elif rew > 0:
        #     rew = min(rew, self.max_reward)
        return rew


    #Currently not working. May be issue with puck collisions happening faster than can be detected with current frame_skip value
    #Helper function to determine when the puck hits something other than mallet 2 for purposes of checking termination
    #Sets self.puck_bounce = True when the puck hits something other than mallet 2 
    def _check_puck_bounced(self):
        if not self.puck_bounce:
             for i in range(self.data.ncon):
                print(self.data.ncon)

                obj_1 = self.model.geom(self.data.contact[i].geom1).name
                obj_2 = self.model.geom(self.data.contact[i].geom2).name
                coll_set = {obj_1, obj_2}
                print(coll_set)
                
                if "mallet2" not in coll_set and "puck" in coll_set:
                    self.puck_bounce = True
                    print("Bounce true")
                    break
