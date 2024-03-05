import importlib
import yaml


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



def get_agent_class_and_config(agent_info : dict):
    """Helper function for getting the class of a specified agent and the associated config.

    Combines the agent name and version to find the required module and then imports the
    agent class within. Uses the path "agents.agent_name.agent_name_version" to import the module. 
    Assumes config path is "agents.agent_name.agent_name_version_config.yml".

    Args:
        agent_info: A dictionary containing the agent's name and version under keys "agent_name"
                    and "version" respectively. Should be the dict under the "agent_info" key 
                    in end_to_end_config.yml.

    Returns:
        A tuple of (agent_class, config_info).
        agent_class: The class of the specified agent. 
        config_info: A dict containing the config information for the specified agent.
    """
    agent_name = agent_info["agent_name"]
    agent_ver = agent_info["version"]

    #Load file that contains agent 
    agent_path = "agents." + agent_name + "." + agent_name + "_" + agent_ver
    agent_mod = importlib.import_module(agent_path)

    #Convert agent path from python to filesystem path and add config suffix
    config_path = agent_path.replace(".", "/") + "_config.yml"
    #Load agent config from file
    config_info = load_yaml(config_path)

    #Get the agent class 
    agent_class = getattr(agent_mod, config_info['class_name'])

    return (agent_class, config_info)


def get_env_class_and_config(env_info):
    """Helper function for getting the class of a specified environment and the associated config.

    Finds the environment with given name and imports the class within. Uses the path "air_hockey_gym.envs.env_name.py" to import the module. 
    Assumes config path is "air_hockey_gym/envs/configs/env_name_config.yml".

    Args:
        env_info: A dictionary containing the environments's name key "env_name"
                    Should be the dict under the "env_info" key in end_to_end_config.yml.

    Returns:
        A tuple of (env_class, config_info).
        env_class: The class of the specified environment. 
        config_info: A dict containing the config information for the specified environment.
    """
    
    env_name = env_info["env_name"]

    #Load file containing environment
    parent_path = "air_hockey_gym.envs."
    env_path = parent_path + env_name
    env_mod = importlib.import_module(env_path)

    #Convert agent path from python to filesystem path and add config suffix
    config_path = "air_hockey_gym/" + parent_path.replace(".", "/") + "configs/" + env_name + "_config.yml"
    #Load config info
    config_info = load_yaml(config_path)

    #Get the env class 
    env_class = getattr(env_mod, config_info["class_name"])

   
    return(env_class, config_info)


def get_test_classes_and_configs(test_info):
    """Helper function for getting the classes of specified validation tests and the associated configs.

    
    Finds the tests with given names and imports the classes within. Uses the path "agent_validation.test_name" to import the module. 
    Assumes config path is "agent_validation/configs/test_name_config.yml".

    Args:
        test_info: A dictionary containing a list of test names under key "test_names"
                    Should be the dict under the "test_info" key in end_to_end_config.yml.

    Returns:
        A list of tuples of (name, test_class, config_info).
        test_name: Name of the specified test. 
        test_class: The class of the specified test. 
        config_info: A dict containing the config information for the specified test.
    """
    
    test_names = test_info["test_names"]
    class_conf_list = []

    for name in test_names:
        #Load file containing environment
        parent_path = "agent_validation."
        test_path = parent_path + name
        test_mod = importlib.import_module(test_path)

        #Convert agent path from python to filesystem path and add config suffix
        config_path = parent_path.replace(".", "/") + "configs/" + name + "_config.yml"
        #Load config info
        config_info = load_yaml(config_path)

        #If a render mode is specified in end_to_end config, set all val test render modes to this value
        if test_info["render_mode"]:
            config_info["test_params"]["render_mode"] = test_info["render_mode"]

        #Get the env class 
        test_class = getattr(test_mod, config_info["class_name"])

        #Add tuple to list of tests
        class_conf_list.append((name, test_class, config_info))

   
    return class_conf_list
