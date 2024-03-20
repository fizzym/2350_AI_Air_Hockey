from air_hockey_gym.envs.air_hockey_base_class_v0 import AirHockeyBaseClass
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

class SingleMalletShootEnvV2(AirHockeyBaseClass):
    """
    V2 Update:
    - Refactor environment to be implement AirHockeyBaseClass
    - Added option to specify discrete or continuous environment

    ### Environment Description
    Air hockey environment used to train agent to play offensively.
    Agent and puck spawn at a fixed location, but this can be customized.

    ### Mallet Definition
	Mallet1 (Mal1) - The controlled mallet. Mallet that plays in -x direction from table coordinate frame.
	The mallet on the left side of the table when looking from default camera angle.

    Mallet 2 (Mal2) - The opponent mallet. Mallet that plays in +x direction from table coordinate frame.
	The mallet on the right side of the table when looking from default camera angle.

	### Observation 
	(12,) numpy array containing positions and velocities of all objects (puck, mallet). This is to facilitate
    input to the RL agent.

	Values:
    (12,) np array (obs)
	The array represents 
        [puck x position, puck y position, puck x velocity, puck y velocity,
        mal1 x position, mal1 y position, mal1 x velocity, mal1 y velocity,
        mal2 x position, mal2 y position, mal2 x velocity, mal2 y velocity]

	Coordinates are in reference frame of controlled mallet
	Mallet reference frame is defined with origin at centre of table, x-y plane in the plane of the table,
	and the positive x-axis pointing away from the controlled mallet's goal. This is done so observation
	is independent of which side of the table the mallet is on. 

    ### Action
    (4,) numpy array (act) containing accelerations of mallet1 and mallet2

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
    
    def __init__(self, max_reward=1, puck_box=[(-0.5,0), (-0.5,0)], mal1_box= [(-0.9,0),(-0.9,0)], mal2_box= [(0.05,0.45),(0.95,-0.45)],
                 accel_mag=1.0, discrete_actions = True, **kwargs):
        """
        All coordinates are in world coordinate frame (center of table, 
        positive x towards opponent goal)
        Inputs:
            max_reward: Maximum reward to return. All rewards are given as percent of this value.
            puck_box: Bounding box which puck will spawn uniformly in.
                [(x1,y1),(x2,y2)] Where (x1,y1) are coordinates of top left corner of box, 
                and (x2,y2) is bottom right
            mal1_box: Bounding box which mallet1 will spawn uniformly in. Same format as puck_box.
            mal2_box: Bounding box which mallet2 will spawn uniformly in. Same format as puck_box.
            accel_mag: The magnitude of acceleration of the agent mallet
            discrete_actions: Flag to set if environment's action space should be discrete or continous.
                Will be continous if False, discrete if True
        """

        super().__init__(
            max_reward=max_reward,
            use_both_agents=False,
            accel_mag = accel_mag,
            discrete_actions=discrete_actions,
            **kwargs,
        )

        #Check box coordinates are well formed i.e. point 1 is above and to the left of point 2
        assert puck_box[0][0] <= puck_box[1][0], "puck_box x coordinates are malformed. Right coordinate is less than left coordinate"
        assert puck_box[0][1] >= puck_box[1][1], "puck_box y coordinates are malformed. Top coordinate is less than bottom coordinate"
        assert mal1_box[0][0] <= mal1_box[1][0], "mal1_box x coordinates are malformed. Right coordinate is less than left coordinate"
        assert mal1_box[0][1] >= mal1_box[1][1], "mal1_box y coordinates are malformed. Top coordinate is less than bottom coordinate"
        assert mal2_box[0][0] <= mal2_box[1][0], "mal2_box x coordinates are malformed. Right coordinate is less than left coordinate"
        assert mal2_box[0][1] >= mal2_box[1][1], "mal2_box y coordinates are malformed. Top coordinate is less than bottom coordinate"

        #Check puck box bounds won't lead to overlapping objects
        #Leave at least 5cm between center of puck and center of table on either side of table
        assert abs(puck_box[0][0]) >= 0.05, "puck_box x1 is less than minimum allowable value of 0.05"

        #Check that mallet 1 box bounds won't lead to overlapping objects
        #Values are limits of mallet playing space plus/minus mallet radius (5cm)
        assert mal1_box[0][0] >= -0.95, "mal_box x1 is less than minimum allowable value of -0.95"
        assert mal1_box[0][1] <= 0.45, "mal_box y1 is greater than maximum allowable value of 0.45"
        assert mal1_box[1][0] <= -0.05, "mal_box x2 is greater than maximum allowable value of -0.05"
        assert mal1_box[1][1] >= -0.45, "mal_box y2 is less than minimum allowable value of -0.45"

        #Check that mallet 2 box bounds won't lead to overlapping objects
        #Values are limits of mallet 2 playing space plus/minus mallet 2 radius (5cm)
        assert mal2_box[0][0] >= 0.05, "mal1_box x1 is less than minimum allowable value of 0.05"
        assert mal2_box[0][1] <= 0.45, "mal1_box y1 is greater than maximum allowable value of 0.45"
        assert mal2_box[1][0] <= 0.95, "mal1_box x2 is greater than maximum allowable value of 0.95"
        assert mal2_box[1][1] >= -0.45, "mal1_box y2 is less than minimum allowable value of -0.45"

        #Store parameters for use in reset function
        self.p_box = puck_box
        self.m1_box = mal1_box
        self.m2_box = mal2_box

        self.discrete_act = discrete_actions
        self.num_steps = 0

    def step(self, a):
        """
        Steps the simulation forward by 1 timestep.

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
    
        #Check if goal scored
        goal_scored, net_scored = self._check_goal_scored()

        #TODO make termination more precise (i.e. remove num_steps limit)
        terminated = goal_scored or self.num_steps > 200

        #TODO determine if truncated condition is needed
        truncated = False

        #TODO define debug info or other useful metrics in info
        info = {}

        #Add rationale for ending episode
        if terminated:
            term_reason = "Goal scored" if goal_scored else "Max steps reached"

            if goal_scored:
                term_reason += " on opponent" if net_scored == 1 else " on agent"

            info["termination_reason"] = term_reason

        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, truncated, info

    
    def reset_model(self):
        """
        Resets the environment including the puck, mallet1, and mallet2
        """
        self.num_steps = 0

        # Spawn puck and agent within desired box
        return self.spawn_in_box(self.m1_box, self.p_box, self.m2_box)
 

    def _get_rew(self, obs):
        """
        Get reward given an observation of the state.

        Inputs:
            obs: Observation of the environment state.
        Outputs:
            reward (float): Reward for current state.
        """
        
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

            if "mallet1" in coll_set and "puck" not in coll_set:
                rew += -0.15 * self.max_reward

        #Reward based on x pos of puck
        rew += 0.001 * self.data.qpos[0] / self.goal_dist * self.max_reward

        #Reward based on x vel of puck when crossing middle
        if abs(self.data.qpos[0] / self.goal_dist) < 0.05 and self.data.qvel[0] > 0:
            rew += 0.5 * self.data.qvel[0] / self.goal_dist * self.max_reward

        return rew
