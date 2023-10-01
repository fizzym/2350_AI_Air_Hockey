from typing import Union
from gymnasium import Env


class RL_Agent:
    """Interface for all RL agents.

    Each type of RL agent (cross-entropy, deep-Q, etc.) should implement this interface.
    Purpose is to enable standardized training and prediction. The user should not need to
    understand anything about the underlying agent implementation other than the parameters
    to edit in the associated config file.
    
    """

    def __init__(self, obs_shape, act_shape, filepath: Union[str, None] = None, **kwargs):
        """Initializes the desired agent with specified arguments.

        Args:
            obs_shape: The shape of the observations the agent will use as inputs.
            act_shape: The shape of the actions that the agent will output. 
            filepath:  Path for loading previously trained agent. If not specified agent will
                        start in default state
        """

        raise NotImplementedError

    def train_agent(self, train_env : Env, **kwargs):
        """Trains agent on specified training environment using specified parameters.

        Args:
            train_env: The Gymnasium environment to use during training.
                       Observation and action shapes should match those specified in constructor.
        
        Returns:
            A training report (i.e. summary and statistics of training progression). 
            TODO: define what form training report will take.
        """
        
        raise NotImplementedError

    def save_agent(self, filepath: str):
        """Saves agent to specified filepath.

        Format of saved agent is specific to each agent and should only be loaded via the associated class.
        Will save agent into directory specified by filepath with name : 'agent_type.filetype'

        Args:
            filepath: The path to the directory to save the agent.

        """
        raise NotImplementedError

    def load_agent(self, filepath: str):
        """Loads a previously trained agent into current object.

        Args:
            filepath: Path for loading previously trained agent.
        """

        raise NotImplementedError

    def predict(self, obs):
        """Given an observation, output an action as predicted by this agent.

        Does not do any training when called, simply predicts off agents current state.

        Args:
            obs: Observation of current environment state. 
                 Must match shape defined in constructor.

        Returns:
            The action that the agent predicts based on the observation.
            Will be in the same shape as act_shape parameter of constructor.

        """

        raise NotImplementedError

