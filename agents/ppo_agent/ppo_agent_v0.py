from agents.rl_agent import RL_Agent
from typing import Union
from gymnasium import Env
from gymnasium.spaces import Discrete

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnMaxEpisodes

class PPO_Agent(RL_Agent):
    """Implementation of the RL_Agent interface using proximal policy optimization.

    Uses Stable Baselines 3's implementation of PPO

    """

    def __init__(self, filepath: Union[str, None] = None, deterministic_predict = False, **kwargs):
        """Initializes the desired agent with specified arguments.

        Args:
            filepath:  Path for loading previously trained agent. If not specified agent will
                        start in default state
            deterministic_predict: Flag to choose whether predictions should be deterministic
        """

        self._net = None
        self.action_space = Discrete(9)
        self.det_predict = deterministic_predict

        if filepath:
            self._net = PPO.load(filepath)

    def train_agent(self, train_env : Env, log_path : str, batch_size : int = 64, max_rew = 10, max_batches : int = 5000, **kwargs):
        """Trains agent on specified training environment using specified parameters.

        Args:
            train_env: The Gymnasium environment to use during training.
                       Observation and action shapes should match those specified in constructor.
            log_path: Directory to save training logs to.
            batch_size: Number of training episodes to play before updating network weights.
            max_rew: The mean reward after which the agent should stop training.
            max_batches: The maximum number of batches to complete before termininating training.
                         Supercedes max reward condition            

        Returns:
            A training report (i.e. summary and statistics of training progression).
        """

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=max_rew, verbose=1)
        eval_callback = EvalCallback(train_env, callback_on_new_best=callback_on_best, verbose=1)
        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_batches, verbose=1)

        callback = CallbackList([eval_callback, callback_max_episodes])

        if not self._net:
            self._net = PPO(MlpPolicy, train_env, batch_size=batch_size, verbose=1, tensorboard_log=log_path)
        else:
            self._net.set_env(train_env)
            self._net.tensorboard_log = log_path
            self._net.batch_size = batch_size

            
        self._net.learn(total_timesteps=(1000*max_batches), tb_log_name='', callback=callback)

        
    def save_agent(self, dir_path: str):
        """Saves agent to specified directory.

        Format of saved agent is specific to each agent and should only be loaded via the associated class.
        Will save agent into directory specified by filepath with name : 'agent_type.filetype'

        Args:
            dir_path: The path to the directory to save the agent.

        """

        filepath = dir_path + "/ppo_agent_v0"
        self._net.save(filepath)

    def load_agent(self, filepath: str):
        """Loads a previously trained agent into current object.

        Args:
            filepath: Path for loading previously trained agent.
        """

        self._net = PPO.load(filepath)

    def predict(self, obs):
        """Given an observation, output an action as predicted by this agent.

        Does not do any training when called, simply predicts off agents current state.
        Prediction is deterministic or not based on constructor argument

        Args:
            obs: Observation of current environment state. 
                 Must match shape defined in constructor.

        Returns:
            The action that the agent predicts based on the observation.
            Will be in the same shape as act_shape parameter of constructor.

        """

        if self._net:
            action, _states = self._net.predict(obs, deterministic=self.det_predict)
            return action
        else:
            return self.action_space.sample()
