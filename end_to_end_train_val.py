from air_hockey_gym.envs import SingleMalletBlockDiscreteEnv
from end_to_end_utils import load_yaml, get_agent_class_and_config, get_env_class_and_config

"""
**********WORK IN PROGRESS - NOT FULLY IMPLEMENTED**********

File for running end-to-end training and validation on an RL model. Currently only includes rough  psuedocode outline of
how file will operate once other components are fully defined.

Reads agent, environment, training, and validation paramaters from config files.
Initializes agents and environments then trains agent and runs validation tests specified.

Creates new directory within 'trained_agents' directory with the format 'agent_name_env_name_date'.
Saves agent, training info, and validation info into 'trained_agents/agent_name_env_name_date'.
"""


if __name__ == '__main__':

    # Load main config file
    main_config = load_yaml("end_to_end_config.yml")
    agent_info = main_config["agent_info"]
    env_info = main_config["env_info"]

    #Get env class and config info
    env_class, env_config = get_env_class_and_config(env_info)

    #Load environment
    env = env_class(**env_config["init_params"])
    print("Environment loaded succesfully.")

    #Get agent class and config info
    agent_class, agent_config = get_agent_class_and_config(agent_info)

    file = agent_info["save_path"]
    #Create agent class with specified params
    #TODO fix hardcoding of observation length
    agent = agent_class(12, len(env.actions), 
                            filepath=file, **agent_config["init_params"])
            
    print("Agent loaded succesfully. Starting training.")
    
    agent.train_agent(env, **agent_config["train_params"])

    print("Agent training completed.")
