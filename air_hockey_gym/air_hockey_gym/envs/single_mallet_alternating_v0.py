from air_hockey_gym.envs.air_hockey_base_class_v0 import AirHockeyBaseClass

import numpy as np
from numpy.random import uniform as unif
from enum import Enum

PUCK_STATIONARY_THRESH = 1e-4
MAX_VEL = 3.0

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.4,
}

class ShotType(Enum):
    STRAIGHT = 0
    BOUNCE_TOP = 1
    BOUNCE_BOTTOM = 2

class SingleMalletAlternatingEnv(AirHockeyBaseClass):
    """
    ###Environment Description

    Modified version of air hockey environment used for training single agent for both offence and defence.
    Alternates between blocking environment and shooting environment initial conditions.
    Controlled mallet spawns randomly in front of it's own goal.
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
    DISCLAIMER: The total accumulated reward is not capped! The max_rew parameter only determines the caling factor used for the rewards.
	
	Value:
	All rewards are given in terms of percent of max_reward argument (default 10)
    "=" indicates reward takes that value and is not changed based on any other conditions
    "+=" indicates if condition is met that amount is added to current reward, in addition to any other conditions met

    Opponent scored on you = -100%
    Scored on opponent = 100%
    Hit anything other than puck = -25%
    Puck position = +0.1% * (puck x position)
    Puck crosses from agent side to opponent side = 500% * (puck x velocity)

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

    
    def __init__(self, max_reward=1, off_def_ratio=[1,1], straight_bounce_ratio=[1,1], max_steps=200,
                 mal1_box_def=[(-0.8,0),(-0.8,0)], mal1_box_off=[(-0.8,0),(-0.8,0)],
                 puck_box_def=[(0.4,0),(0.4,0)], puck_box_off=[(-0.4,0),(-0.4,0)], puck_drift_vel_range=[0,0.2],
                 mal2_puck_dist_range=[0.25,0.25], mal2_vel_range=[1,1], mal2_box_off=[(0.9,0.4),(0.9,0.4)],
                 max_accel=5, discrete_actions = True, **kwargs):
                 """
                 All coordinates are in world coordinate frame (center of table, 
                 positive x towards opponent goal)

                 Inputs:
                    max_reward: Maximum reward to return PER TIME STEP. All rewards are given as percent of this value.
                            Note that agents can accumulate more than this during a single episode.
                    off_def_ratio: Ratio of offensive to defensive simulations to run. Runs only a single type if one element is 0.
                    straight_bounce_ratio: Ratio of straight to bounce shots during defensive training.
                    max_steps: Maximum simulation steps to wait before forcefully terminating.
                    mal1_box_def: Bounding box which mallet1 will spawn uniformly in. (DEFENCE)
                            [(x1,y1),(x2,y2)] Where (x1,y1) are coordinates of top left corner of box, 
                            and (x2,y2) is bottom right
                    mal1_box_off: Bounding box which mallet1 will spawn uniformly in. (OFFENCE)
                    puck_box_def: Bounding box which puck will spawn uniformly in. (DEFENCE)
                    puck_box_off: Bounding box which puck will spawn uniformly in. (OFFENCE)
                    puck_drift_vel_range: Magnitude range of drift velocities for puck. (OFFENCE)
                    mal2_puck_dist_range: Absolute value of range of distances between mallet 2 and the puck. (DEFENCE)
                    mal2_vel: Absolute value of range of velocites mallet 2 can spawn with. (DEFENCE)
                    mal2_box_off: Bounding box which mallet2 will spawn uniformly in. (OFFENCE)
                    max_accel: The max acceleration of the agent mallet
                    discrete_actions: Flag to set if environment's action space should be discrete or continous.
                              Will be continous if False, discrete if True
        
                 """
                 super().__init__(max_reward=max_reward, use_both_agents=False, max_accel=max_accel,
                                  discrete_actions=discrete_actions, **kwargs)
       

                 #Store parameters for use in reset function
                 self.off_def_ratio = off_def_ratio
                 self.straight_bounce_ratio = straight_bounce_ratio
                 self.m1_box_def = mal1_box_def
                 self.m1_box_off = mal1_box_off
                 self.p_box_def = puck_box_def
                 self.p_box_off = puck_box_off
                 self.p_drift_vel = puck_drift_vel_range
                 self.m2_puck_dist = mal2_puck_dist_range
                 self.m2_vel = mal2_vel_range
                 self.m2_box_off = mal2_box_off
                
                 self.discrete_act = discrete_actions
                 self.max_steps = max_steps
                 #Initialize step counter
                 self.num_steps = 0
                 #Create alternating def/off flag
                 self.off_flag = True
                 self.mode_counter = 0
                 #Create bounce cycle counter
                 self.bounce_flag = False
                 self.bounce_counter = 0
                 self.bounce_top_flag = True


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

        ob = self.step_sim(np.multiply(self.max_accel, action))

        self.num_steps += 1

        reward = self._get_rew(ob)

        info = {}
    
        #Check if goal scored
        goal_scored, net_scored = self._check_goal_scored()
    
        #Set termination
        terminated = goal_scored or self.num_steps > self.max_steps

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
        """Resets the environment for the next episode
        """

        self.num_steps = 0
        if self.get_next_mode() == "off":
            return self.reset_off()
        else:
            return self.reset_def()
        
    def get_next_mode(self):
        """
            Determines if next episode should be offensive or defensive according to off_def_ratio.
            If either array element is 0, only selects available mode for whole training run.
        """

        if self.off_def_ratio[0] == 0:
            return "def"
        elif self.off_def_ratio[1] == 0:
            return "off"
        else:
            self.mode_counter += 1
            if self.off_flag:
                if self.mode_counter >= self.off_def_ratio[0]:
                    self.mode_counter = 0
                    self.off_flag = False
                return "off"
            else:
                if self.mode_counter >= self.off_def_ratio[1]:
                    self.mode_counter = 0
                    self.off_flag = True
                return "def"
    
    def reset_def(self):
        """Resets the environment for a defensive episode
        """

        return self.reset_bounce_def(self.get_next_shot_mode())
    
    def reset_off(self):
        """Resets the environment for an offensive episode
        """

        # Spawn puck and agent within desired box
        temp_obs = self.spawn_in_box(self.m1_box_off, self.p_box_off, self.m2_box_off)

        # Pick out coordinates
        puck_x = temp_obs[0]
        puck_y = temp_obs[1]

        m1_x = temp_obs[4]
        m1_y = temp_obs[5]

        m2_x = temp_obs[8]
        m2_y = temp_obs[9]

        # Selects random drift velocity within range and random direction between 0 and 2pi
        p_vel_mag = unif(self.p_drift_vel[0], self.p_drift_vel[1])
        p_vel_angle = unif(0, 2 * np.pi)
        p_vel = p_vel_mag * np.array([np.cos(p_vel_angle), np.sin(p_vel_angle)])
        
        pos = [puck_x, puck_y, m1_x, m1_y, m2_x, m2_y]
        vel = [p_vel[0],p_vel[1],0,0,0,0]

        return self.set_custom_state(pos,vel) 
    
    def get_next_shot_mode(self):
        """
            Determines if next defensive episode should initialize the opponent for a straight shot or bounce shot according to straight_bounce_ratio.
            If either array element is 0, only selects available mode for whole training run.
        """

        if self.straight_bounce_ratio[0] == 0:
            return self.get_next_bounce_mode()
        elif self.straight_bounce_ratio[1] == 0:
            return ShotType.STRAIGHT
        else:
            self.bounce_counter += 1
            if self.bounce_flag:
                if self.bounce_counter >= self.straight_bounce_ratio[1]:
                    self.bounce_counter = 0
                    self.bounce_flag = False
                return self.get_next_bounce_mode()
            else:
                if self.bounce_counter >= self.straight_bounce_ratio[0]:
                    self.bounce_counter = 0
                    self.bounce_flag = True
                return ShotType.STRAIGHT
            
    def get_next_bounce_mode(self):
        """Alternates between bounce shots from the top and bounce shots from the bottom
        """

        if self.bounce_top_flag:
            self.bounce_top_flag = False
            return ShotType.BOUNCE_TOP
        else:
            self.bounce_top_flag = True
            return ShotType.BOUNCE_BOTTOM

    def reset_bounce_def(self, shot_type: ShotType = ShotType.STRAIGHT):
        """

        Args:
            shot_type: Enum class with 3 options (STRAIGHT, BOUNCE_TOP, BOUNCE_BOTTOM)
        """
        #Spawn puck and agent within desired box
        temp_obs = self.spawn_in_box(self.m1_box_def, self.p_box_def, [(0,0), (0,0)])

        #Pick out coordinates
        puck_x = temp_obs[0]
        puck_y = temp_obs[1]

        m1_x = temp_obs[4]
        m1_y = temp_obs[5]

        #Position of goal is regular at bounce_mode == 0
        proj_goal_pos = np.array([-self.goal_dist, 0])

        #Flip y coord if bouncing off bottom edge
        if shot_type == ShotType.BOUNCE_TOP:
            proj_goal_pos[1] = 2 * self.table_width
        elif shot_type == ShotType.BOUNCE_BOTTOM:
            proj_goal_pos[1] = - 2 * self.table_width

        #Vector from puck position to reflected goal 
        bounce_vec = proj_goal_pos - temp_obs[0:2]
        
        #Pick mallet 2 velocity
        m2_vel_mag = unif(self.m2_vel[0], self.m2_vel[1])
        mal2_vel = m2_vel_mag * bounce_vec / np.linalg.norm(bounce_vec)

        #Pick distance mallet 2 starts behind puck
        m2_puck_dist = unif(self.m2_puck_dist[0], self.m2_puck_dist[1])
        m2_pos = temp_obs[0:2] - m2_puck_dist * (bounce_vec / np.linalg.norm(bounce_vec))

        pos = [puck_x, puck_y, m1_x, m1_y, m2_pos[0], m2_pos[1]]
        vel = [0,0,0,0, mal2_vel[0],mal2_vel[1]]

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
                rew += -0.1 * self.max_reward

        #Reward based on x pos of puck
        rew += 0.001 * self.data.qpos[0] / self.goal_dist * self.max_reward

        #Reward based on x vel of puck when crossing middle
        if abs(self.data.qpos[0] / self.goal_dist) < 0.05 and self.data.qvel[0] > 0:
            angle_factor = 0.25
            vel_factor = self.data.qvel[0] + angle_factor * abs(self.data.qvel[1])
            rew += 0.5 * vel_factor / self.goal_dist * self.max_reward

        return rew
