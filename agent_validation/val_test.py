from agents.rl_agent import RL_Agent

class ValidationTest():
    """Interface for all validation test.
    
    """

    def __init__(self, **kwargs):
        """Perform any required initialization.
        """
    
        raise NotImplementedError

    def test_agent(self, agent : RL_Agent,  log_path : str, **kwargs):
        """Perform desired test. Saves test info to specified log_path.

        Args:
            agent: The agent to test.
            log_path: Directory to save test logs to.

        """
        