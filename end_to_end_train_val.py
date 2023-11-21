from end_to_end_utils import load_yaml, get_agent_class_and_config, get_env_class_and_config, get_test_class_and_config

import os
import datetime
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


if __name__ == '__main__':


    # Load main config file
    main_config = load_yaml("end_to_end_config.yml")
    agent_info = main_config["agent_info"]
    env_info = main_config["env_info"]
    test_info = main_config["test_info"]

    #Get env class and config info
    env_class, env_config = get_env_class_and_config(env_info)

    #Load environment
    env = env_class(**env_config["init_params"])
    print("Environment loaded succesfully.")

    #Get agent class and config info
    agent_class, agent_config = get_agent_class_and_config(agent_info)

    file = agent_info["load_path"]
    #Create agent class with specified params
    #TODO fix hardcoding of observation length
    agent = agent_class(12, len(env.actions), 
                            filepath=file, **agent_config["init_params"])
            
    print("Agent loaded succesfully. Starting training.")

    #Get current date-time and format
    cur_datetime = datetime.datetime.now()
    formatted_datetime = cur_datetime.strftime("%m-%d-%H-%M")

    #Make directory for storing information related to agent training and evaluation
    save_path = "trained_models/" + agent_info["agent_name"] + "_" + agent_info["version"] 
    save_path += "_" + env_info["env_name"] + "_" + formatted_datetime

    os.mkdir(save_path)

    #Make directory for training info in parent directory
    train_path = save_path + "/training_info"
    os.mkdir(train_path)

    #Make directory for storing config file copies
    config_save_path = save_path + "/configs"
    os.mkdir(config_save_path)

    #Make directory for storing validation data
    val_path = save_path + "/val_info"
    os.mkdir(val_path)

    yamls = ["end_to_end_config", "agent_config", "env_config"]
    configs = [main_config, agent_config, env_config]

    # Write copies of config files into end-to-end directory
    for i in range(0,3):
        file = config_save_path + "/" + yamls[i] + ".yml"
        with open(file, 'w') as yaml_file:
            yaml.dump(configs[i], yaml_file, default_flow_style=False)
    
    agent.train_agent(env, train_path, **agent_config["train_params"])
    print("Agent training completed.")

    agent.save_agent(save_path)

    #Get test class and config info
    test_class, test_config = get_test_class_and_config(test_info)

    #Load test
    val_test = test_class()

    print("Testing agent.")

    val_test.test_agent(agent, val_path, **test_config["test_params"])

    
