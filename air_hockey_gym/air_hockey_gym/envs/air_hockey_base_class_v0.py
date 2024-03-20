from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Discrete, Dict
import os
import mujoco

import numpy as np
from numpy.random import uniform as unif

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.4,
}

class AirHockeyBaseClass(MujocoEnv):
    """Base class which all single mallet air hockey training environments should inherit. 
    
    Implements the following common functionality:
        __init__ - initializes underlying simulation and stores max_reward
        step_sim - runs a single simulation step for a given set of accelerations 
                   (agent and opponent) and returns observation
        reset_model - resets simulation to default posiitons
        set_custom_state - sets state of simulation (puck and mallet positions and velocities)
        spawn_in_box - spawns pucks and mallets randomly within defined boxes with zero velocity
        _check_goal_scored - Checks if a goal has been scored and if so which net was it on
        _get_obs - Returns observation of system

    When making new training environments, the following functions should be implemented:
        step(action) - takes agent action, steps simulation and returns 
                       (observation, reward, terminated, info)
        _get_rew - method to calculate reward for given environment step
        Desired modifications to already implemented methods

    
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

    def __init__(self, max_reward=10, use_both_agents = False, discrete_actions = False, 
                accel_mag = 1, **kwargs):
        """Initializes environment and underlying MuJoCo simulation

        Inputs:
            max_reward: The max reward to be given during training. All rewards are a given as a
                        fraction of this reward.
            use_both_agents: Flag to set number of agents in environment. If False, only left mallet is used as agent.
            discrete_actions: Flag to set if environment's action space should be discrete or continous.
                              Will be continous if False, discrete if True
            accel_mag: The magnitude of acceleration if action space is discrete
        """

        self.use_both_agents = use_both_agents
        #Store max reward
        self.max_reward = max_reward

        #Distance of goal line from center of table
        self.goal_dist = 1.0
        self.goal_width = 0.13

        #Width of table
        self.table_width = 0.5

        #Define observation space
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)

        #Modify observation space for two agents
        if use_both_agents:
            #Name of agents used as keys in dictionaries
            self.mal1_name = "mal1"
            self.mal2_name = "mal2"
            observation_space = Dict({self.mal1_name : observation_space, self.mal2_name : observation_space})
        
        self.asset_path = os.path.join(os.path.dirname(__file__), "assets/")

        MujocoEnv.__init__(
            self,
            self.asset_path + "table_2_mallets.xml",
            #Defines how many time steps should be executed between each step function call 
            frame_skip=40,
            observation_space=observation_space,
            **kwargs,
        )

        if discrete_actions:
            #Set up discrete actions
            #Initialize 0 vector and 8 allowable directions
            self.actions = accel_mag * np.array([[0.0,0.0],
                            [1.0,0.0],
                            [0.707,0.707],
                            [0.0,1.0],
                            [-0.707,0.707],
                            [-1.0,0.0],
                            [-0.707,-0.707],
                            [0.0,-1.0],
                            [0.707,-0.707]])
            self.action_space = Discrete(len(self.actions))

        #Modify action space for two agents
        if use_both_agents:
            self.action_space = Dict({self.mal1_name : self.action_space, self.mal2_name : self.action_space})

    def step_sim(self, agent_action, opp_action = [0,0]):
        """Runs a single simulation step with given action.

        Inputs:
            agent_action: (2,) numpy array containing accelerations of agent mallet
            opp_action: (2,) numpy array containing accelerations of opponent mallet

        Returns:
            observation: Observation of simulation after action has been takem
        """
        self.do_simulation(np.concatenate((agent_action, opp_action)), self.frame_skip)
        return self._get_obs()


    def reset_model(self):
        """Resets simulation such that all objects are reset to default positions and velocities.

        Returns:
            Observation of simulation after reset is performed.
        """
        
        #Reset all values to default from XML
        qpos = np.zeros(13,)
        qvel = np.zeros(13,)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def set_custom_state(self, pos, vel):
        """Set state of system (puck and mallet positions and velocities).
        Inputs:
            pos: Length 6 list-like which contains desired positions of objects in left mallet's reference frame.
                 Format is [puck_x, puck_y, agent_mallet_x, agent_mallet_y, opp_mall_x, opp_mall_y]
            vel: Length 6 list-like which contains desired velocities of objects in left mallet's reference frame.
                 Same format as pos, but for velocity components.
        Returns:
            obs: Current observation of system after desired state is set. 
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

    def spawn_in_box(self, agent_box, puck_box, opp_box):
        """Spawns mallets and pucks randomly within their own defined box.

            All boxes are defined as [(x1,y1),(x2,y2)] Where (x1,y1) are coordinates of top left
            corner of box, and (x2,y2) is bottom right, in world coordinate frame
            (center of table, positive x towards opponent goal).

            Inputs:
                agent_box: Box that agent mallet will randomly be spawned in.
                puck_box: Box that puck will randomly be spawned in.
                opponent_box: Box that opponent mallet will randomly be spawned in.

            Returns:
                observation: Observation of simulation after random spawning.
        """

        coord_gen = lambda box : [unif(box[0][0], box[1][0]), unif(box[0][1], box[1][1])]

        agent_coords = coord_gen(agent_box)
        puck_coords = coord_gen(puck_box)
        opp_coords = coord_gen(opp_box)

        pos = puck_coords + agent_coords + opp_coords
        vel = np.zeros(6,)

        return self.set_custom_state(pos, vel)

    
    def _check_goal_scored(self) -> (bool, int):
        """Checks if a goal hasa been scored with the current system state.

        Returns:
            Tuple of (goal_scored, net_scored on).

            goal_scores is a bool, which is true if a goal has been scored.
            net_scored on is an int which represents which net has been scored on.
                0 - No goal scored
                1 - Goal scored on opponent
                2 - Goal scored on agent
        """
        goal_scored = np.abs(self.data.qpos[0]) > self.goal_dist and np.abs(self.data.qpos[1]) < self.goal_width

        net_scored_on = 0

        if goal_scored:
            net_scored_on = 1 if self.data.qpos[0] > 0 else 2

        return (goal_scored, net_scored_on)

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

        obs = np.concatenate([cart_p, cart_m1, cart_m2])

        if self.use_both_agents:
            m1_obs = np.copy(obs)
            #Mallet 2 reference frame is 180 rotation of Mallet 1 frame, therefore multiply all values by -1
            m2_obs = -1 * np.copy(obs)
            #Switch ordering of values in M2 obs so that controlled mallet is still indices 4-8 and 
            #opponent mallet is indices 8-12
            m2_obs[4:8] = -1 * obs[8:12]
            m2_obs[8:12] = -1 * obs[4:8]

            obs = {self.mal1_name: m1_obs, self.mal2_name : m2_obs}

        return obs

    def _get_rew(self):
        raise NotImplementedError 
