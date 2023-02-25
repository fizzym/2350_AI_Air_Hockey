from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": -1,
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
        "render_fps": 250,
    }

    def __init__(self, **kwargs):

        #TODO define max speeds and positions
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            "/home/bradychurch/Desktop/air_hockey_gym/air_hockey_gym/envs/assets/air_hockey.xml",
            #TODO check if I need to change frame skip
            2,
            observation_space=observation_space,
            #TODO figure out how to define camera config
            #default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

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