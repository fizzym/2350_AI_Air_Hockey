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
    "distance": 2,
    "lookat": np.array((0, 0, 0.1)),
    "elevation": -90,
}

#TODO see if I need to add EzPickle
class SingleMalletShootDiscreteEnv(MujocoEnv):
    """

    ###Environment Description
    Air hockey environment used to train agent to play offensively.
    Agent and puck spawn at a fixed location, but this can be customized.

    ### Mallet Definition
	Mallet (Mal) - The controlled mallet. Mallet that plays in -x direction from table coordinate frame.
	The mallet on the left side of the table when looking from default camera angle. 

	### Observation 
	(8,) numpy array containing positions and velocities of all objects (puck, mallet). This is to facilitate
    input to the RL agent.

	Values:
    (8,) np array (obs)
	The array represents [puck x position, puck y position, puck x velocity, puck y velocity,
        mal x position, mal y position, mal x velocity, mal y velocity].

	Coordinates are in reference frame of controlled mallet
	Mallet reference frame is defined with origin at centre of table, x-y plane in the plane of the table,
	and the positive x-axis pointing away from the controlled mallet's goal. This is done so observation
	is independent of which side of the table the mallet is on. 

    ### Action
    (2,) numpy array (act) containing accelerations of mallet

    act[:2] = [x1 acceleration, y1 accleration] acceleration of Mallet in Mallet reference frame

	### Reward
	Dict of mallet name to reward (int)

	Value:
	All rewards are given in terms of percent of max_reward argument (default 10)
    "=" indicates reward takes that value and is not changed based on any other conditions
    "+=" indicates if condition is met that amount is added to current reward, in addition to any other conditions met

    You score on opponent = +100%
    Opponent scored on you = -100%
    Hit puck += 10%
    Hit anything other than puck -= 50%
    

	### Episode End
	A training episode terminates when the any of the following conditions are met:
    - Centre point of the puck passes the goal line on either side
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
    #       In v0 of the environment, the spawn position is fixed to (-0.5,0) by default

    #@param mal_box Bounding box which mallet will spawn uniformly in.
    #       Format:[(x1,y1),(x2,y2)] Where (x1,y1) are coordinates of top left corner of box, and (x2,y2) is bottom right
    #       In world coordinate frame (center of table, positive x towards opponent goal)
    #       In v0 of the environement, the spawn position is fixed to (-0.9,0) by default
    
    def __init__(self, max_reward=10, puck_box=[(-0.5,0), (-0.5,0)], mal_box= [(-0.9,0),(-0.9,0)], max_accel=5, **kwargs):

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
        assert mal_box[0][0] <= mal_box[1][0], "mal_box x coordinates are malformed. Right coordinate is less than left coordinate"
        assert mal_box[0][1] >= mal_box[1][1], "mal_box y coordinates are malformed. Top coordinate is less than bottom coordinate"

        #Check puck box bounds won't lead to overlapping objects
        #Leave at least 5cm between center of puck and center of table on either side of table
        assert abs(puck_box[0][0]) >= 0.05, "puck_box x1 is less than minimum allowable value of 0.05"

        #Check that mallet box bounds won't lead to overlapping objects
        #Values are limits of mallet playing space plus/minus mallet radius (5cm)
        assert mal_box[0][0] >= -0.95, "mal_box x1 is less than minimum allowable value of -0.95"
        assert mal_box[0][1] <= 0.45, "mal_box y1 is greater than maxinyn allowable value of 0.45"
        assert mal_box[1][0] <= -0.05, "mal_box x2 is greater than maximum allowable value of -0.05"
        assert mal_box[1][1] >= -0.45, "mal_box y2 is less than minimum allowable value of -0.45"

        #Store parameters for use in reset function
        self.p_box = puck_box
        self.m1_box = mal_box

        #Distance of goal line from center of table
        self.goal_dist = 1.0
        self.goal_width = 0.13

        self.num_steps = 0
        self.puck_bounce = False

        #Note: Other MuJoCo Envs do not seem to define limits to observation space
        #The limits don't seem to be checked/enforced anywhere so I have also not included them
        observation_space = Box(low=-np.inf, high=np.inf, shape=(2,4), dtype=np.float64)
        
        self.asset_path = os.path.join(os.path.dirname(__file__), "assets/")

        MujocoEnv.__init__(
            self,
            #Using table with only 1 mallet
            self.asset_path + "table_1_mallet.xml",
            #Defines how many time steps should be executed between each step function call 
            frame_skip=40,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    #@param a - number from 0-8
    #See documentation above
    def step(self, a=0):

        action = list(self.actions[a])
        # action.append(0)
        # action.append(0)

        self.do_simulation(np.multiply(self.max_accel,action), self.frame_skip)
        ob = self._get_obs()

        self.num_steps += 1

        reward = self._get_rew(ob)
    
        #Check if centre of puck is in goal
        #Puck x position is first defined joint, puck y position is 2nd defined joint 
        #and (0,0) at center of table
        #Goal line is 1m in x from center and goal is 26 cm wide
        goal_scored = np.abs(self.data.qpos[0]) > self.goal_dist and np.abs(self.data.qpos[1]) < self.goal_width

        #TODO make termination more precise (i.e. remove num_steps limit)

        terminated = goal_scored or self.num_steps > 200

        #TODO determine if truncated condition is needed
        truncated = False

        #TODO define debug info or other useful metrics in info
        info = {}

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
        #Pick mallet start location within defined box
        m1_x = unif(self.m1_box[0][0], self.m1_box[1][0]) + 0.25
        m1_y = unif(self.m1_box[0][1], self.m1_box[1][1])

        #Start with inital values for qpos and qvel
        qpos = self.init_qpos
        qvel = self.init_qvel

        #Set all desired coordinates
        qpos[0] = puck_x
        qpos[1] = puck_y

        qpos[3] = m1_x
        qpos[4] = m1_y

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):

        #Calculate current Cartesian coordinates
        mujoco.mj_kinematics(self.model,self.data)

        #Note: All frames have same orientation, but different origins. Therefore qvel values
        #are still in cartesian coords

        #Puck's DOF are defined 1st and 2nd in the XML
        cart_p = np.concatenate((self.data.geom("puck").xpos[:2], self.data.qvel[0:2]))
    	#Mallet's DOF are defined 4th and 5th in the XML
        cart_m1 = np.concatenate((self.data.geom("mallet1").xpos[:2], self.data.qvel[3:5]))
        
        m1_obs = np.concatenate((cart_p, cart_m1))

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
