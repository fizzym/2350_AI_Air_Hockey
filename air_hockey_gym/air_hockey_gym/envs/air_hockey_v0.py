from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
import os
import mujoco

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



    def __init__(self, **kwargs):

    	#Name of agents used as keys in dictionaries
        self.mal1_name = "mal1"
        self.mal2_name = "mal2"

        #TODO define max speeds and position
        observation_space = Dict({self.mal1_name: Box(low=-np.inf, high=np.inf, shape=(3,4), dtype=np.float64),
        self.mal2_name: Box(low=-np.inf, high=np.inf, shape=(3,4), dtype=np.float64)})
        
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

    def _initialize_simulation(self):

        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)
        
        #Not sure why, but model will not render without calling this
        mujoco.mj_forward(self.model, self.data)
      

    #@param a - action
    def step(self, a):
        #TODO define reward function 
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        #TODO define terminated condition
        terminated = False
        #TODO determine if truncated condition is needed
        truncated = False
        #TODO define debug info or other useful metrics in info
        info = {}
        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, truncated, info

    
    def reset_model(self):
        
        #TODO reset model
        self.set_state()
        return self._get_obs()

    def _get_obs(self):

        #TODO access observation from self.data
        array1 = np.ones((3,4), np.float64)
        array2 = np.zeros((3,4), np.float64)
        return {self.mal1_name: array1, self.mal2_name : array2}