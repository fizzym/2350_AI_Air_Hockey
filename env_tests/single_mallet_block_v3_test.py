import gymnasium as gym
from air_hockey_gym.envs.single_mallet_blocking_v3 import SingleMalletBlockEnv
import time
import numpy as np

max_rew = 5.0
env = SingleMalletBlockEnv(max_reward=max_rew, render_mode="human")

#File to test SingleMalletBlockEnv


obs, _ = env.reset()

assert len(obs) == 12


#Test setting_custom state

pos = [-0.2, 0.19, -0.3, 0.13, 0.3, 0.14]
vel = [0.5, 0.1, 0.2, -0.2, -0.75, 0.12]

expected_state = [-0.2, 0.19, 0.5, 0.1, -0.3, 0.13, 0.2, -0.2, 0.3, 0.14, -0.75, 0.12]

obs = env.set_custom_state(pos, vel)
assert np.allclose(expected_state, obs), f"Expected observation to be {expected_state}. Was instead {obs}."


#Test termination in opponent goal
pos = [0,0, -0.3, -0.15, 0.3, 0.15]
vel = np.zeros((6,))
vel[0] = 0.75

env.set_custom_state(pos, vel)

action = 0
terminated_correctly = False
for i in range(100):

    obs, rew, term, _, info_dict = env.step(action)

    if(term):
        terminated_correctly = True
        assert rew == max_rew, f"Expected reward to be {max_rew}. Instead was {rew}."
        assert info_dict["termination_reason"] == "Goal scored on opponent", f"Incorrect termination reason"
        break

assert terminated_correctly

#Test termination in agent goal
vel[0]= -vel[0]
env.set_custom_state(pos, vel)
action = 0
terminated_correctly = False
for i in range(100):

    obs, rew, term, _, info_dict = env.step(action)

    if(term):
        terminated_correctly = True
        assert rew == -max_rew, f"Expected reward to be {max_rew}. Instead was {rew}."
        assert info_dict["termination_reason"] == "Goal scored on agent", f"Incorrect termination reason"
        break

assert terminated_correctly

env.close()
print("\nTests Passed.\n")
