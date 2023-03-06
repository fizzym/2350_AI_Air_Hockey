from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import mujoco

import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.4,
}

#TODO see if I need to add EzPickle
class AirHockeyEnv(MujocoEnv):

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

        #TODO define max speeds and positions
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        self.asset_path = os.path.join(os.path.dirname(__file__), "assets/")
        print(self.asset_path)

        MujocoEnv.__init__(
            self,
            self.asset_path + "table.xml",
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
        return np.ones((12,), np.float64)