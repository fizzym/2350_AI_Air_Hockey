from air_hockey_gym.envs import SingleMalletBlockDiscreteEnv
import importlib
import yaml

"""
**********WORK IN PROGRESS - NOT FULLY IMPLEMENTED**********

File for running end-to-end training and validation on an RL model. Currently only includes rough  psuedocode outline of
how file will operate once other components are fully defined.

Reads agent, environment, training, and validation paramaters from config files.
Initializes agents and environments then trains agent and runs validation tests specified.

Creates new directory within 'trained_agents' directory with the format 'agent_name_env_name_date'.
Saves agent, training info, and validation info into 'trained_agents/agent_name_env_name_date'.
"""


    

def load_yaml(filepath) -> dict:
    """Helper function to load a yaml file into a dictionary given a filepath.
    
    Args:
        filepath: Path to the yaml file to load.

    Returns:
        The contents of the yaml file as a dictionary.
    """
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    
    return config



def get_agent_class_and_config(agent_info):
    """Helper function for getting the class of a specified agent and the path to the associated config.

    Combines the agent name and version to find the required module and then imports the
    agent class within. Uses the path "agents.agent_name.agent_name_version" to import the module. 
    Assumes config path is "agents.agent_name.agent_name_version_config.yml".

    Args:
        agent_info: A dictionary containing the agent's name and version under keys "agent_name"
                    and "version" respectively. Should be the dict under the "agent_info" key 
                    in end_to_end_config.yml.

    Returns:
        A tuple of (agent_class, config_path).
        agent_class: The class of the specified agent. 
        config_path: The path to the specified agent's config file.

    """

    agent_name = agent_info["agent_name"]
    agent_ver = agent_info["version"]

    #Load file that contains agent 
    agent_path = "agents." + agent_name + "." + agent_name + "_" + agent_ver
    agent_mod = importlib.import_module(agent_path)

    #Set name of agent class
    #(Is probably possible to also determine this dynamically, but we may need
    # to implement some more strict naming standards.)
    if agent_name == "x_entropy_agent":
        class_name = "X_Entropy_Agent"
    
    #Get the agent class 
    agent_class = getattr(agent_mod, class_name)

    #Convert agent path from python to filesystem path and add config suffix
    config_path = agent_path.replace(".", "/") + "_config.yml"

    return (agent_class, config_path) 

    # Verify loaded params
    main_params = config['main_params']
    print(main_params)

    agent_params = config['agent_params'][main_params['agent']]
    print(agent_params)

if __name__ == '__main__':

    # Load main config file
    main_config = load_yaml("end_to_end_config.yml")
    agent_info = main_config["agent_info"]

    #Load environment
    # TODO implement proper environment loading. 
    env = SingleMalletBlockDiscreteEnv(max_reward=20, render_mode="rgb_array", 
                                       mal2_vel_range = [0.5,1.5], max_accel=2.5)
    print("Environment loaded succesfully.")

    #Get agent class and config path
    agent_class, agent_config_path = get_agent_class_and_config(agent_info)
    #Load agent's config information
    agent_config = load_yaml(agent_config_path)

    file = agent_info["save_path"]
    #Create agent class with specified params
    #TODO fix hardcoding of observation length
    agent = agent_class(12, len(env.actions), 
                            filepath=file, **agent_config["init_params"])
            
    print("Agent loaded succesfully. Starting training.")
    
    agent.train_agent(env, **agent_config["train_params"])

    print("Agent training completed.")
