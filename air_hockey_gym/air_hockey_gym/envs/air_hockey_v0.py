from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import mujoco
import mediapy as media

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
        "render_fps": 250,
    }

    def __init__(self, **kwargs):

        #TODO define max speeds and positions
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        self.asset_path = "/home/bradychurch/Desktop/459_ai_air_hockey_gym/air_hockey_gym/air_hockey_gym/envs/assets/"

        asset_names = []
        for file in os.listdir(self.asset_path):
        	if file.endswith(".stl"):
        		asset_names.append(file)

        print(asset_names)
        self.assets = {}
        for file in asset_names:
            with open(self.asset_path+file, 'rb') as f:
                self.assets[file] = f.read()

        print(self.assets.keys())
        MujocoEnv.__init__(
            self,
            self.asset_path + "air_hockey.xml",
            #TODO check if I need to change frame skip
            2,
            observation_space=observation_space,
            #TODO figure out how to define camera config
            #default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        #self._initialize_simulation()

    def _initialize_simulation(self):
        print("In air hockey _initialize_simulation")

        xml = """
		<mujoco>
		  <worldbody>
		    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
		    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
		  </worldbody>
		</mujoco>
		"""
        self.model = mujoco.MjModel.from_xml_path(self.fullpath, self.assets)
        print(self.model.nu)
    	#self.model = mujoco.MjModel.from_xml_string(xml)
    	# MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)
        

        renderer = mujoco.Renderer(self.model)
        mujoco.mj_forward(self.model, self.data)
        renderer.update_scene(self.data)
        media.write_image(self.asset_path + "test_image.png", renderer.render())

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