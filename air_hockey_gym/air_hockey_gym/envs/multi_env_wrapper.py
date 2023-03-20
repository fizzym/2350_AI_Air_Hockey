from copy import deepcopy
import numpy as np
from gymnasium.spaces import Box


class MultiEnvWrapper:
    """
    Class to represent a group of parallel running environments
    """
    def __init__(self, env, num_envs: int = 1, **kwargs):
        self.num_envs = num_envs
        self.envs = np.empty(num_envs, dtype=env)
        self.action_space = np.empty(num_envs, dtype=Box)

        for i in range(num_envs):
            self.envs[i] = env(**kwargs)
            self.action_space[i] = self.envs[i].action_space

    # Step all environments one-by-one
    # Note: step() automatically resets an environment when terminated while it is mass-stepping
    # for efficiency purposes. To have more control, use the overloaded step() below
    def step(self, a):
        assert (
            len(a) == self.num_envs
        ), f"Expected Actions array of length {self.num_envs}. Instead was {len(a)}."

        ret_val = []
        for i in range(self.num_envs):
            ob, reward, terminated, truncated, info = self.envs[i].step(a[i])
            ret_val.append((ob, reward, terminated, truncated, info))

            # Debugging Messages
            # print("Obs of mallet1, Env " + str(i) + ": " + str(ob["mal1"]))
            # print("Obs of mallet2, Env " + str(i) + ": " + str(ob["mal2"]))
            # print()

            if terminated:
                self.envs[i].reset()

        return ret_val

    # Steps environment at index
    def step_at(self, a, index):
        return self.envs[index].step(a)

    # Resets all environments one-by-one
    def reset(self):
        ret_val = []
        for i in range(self.num_envs):
            obs, _ = self.envs[i].reset()
            ret_val.append((obs, _))

        return ret_val

    # Resets environment at index
    def reset_at(self, index):
        return self.envs[index].reset()

    # Resets all models one-by-one
    def reset_model(self):
        ret_val = []
        for i in range(self.num_envs):
            ret_val.append(self.envs[i].reset_model())

        return ret_val

    # Resets model at index
    def reset_model_at(self, index):
        return self.envs[index].reset_model()

    def close(self):
        ret_val = []
        for i in range(self.num_envs):
            ret_val.append(self.envs[i].close())

        return ret_val

    def close_at(self, index):
        return self.envs[index].close()

    # Return an array of samples for each environment
    def action_space_sample(self):
        actions = []
        for i in range(self.num_envs):
            actions.append(self.action_space[i].sample())

        return actions
