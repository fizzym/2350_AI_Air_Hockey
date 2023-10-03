from agents.rl_agent import RL_Agent
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
    """
    
    main_params = load_config(main_config_path)

    agent = find_agent_from_name(main_params[agent_name])
    env = find_env_from_name(main_params[env_name])

    save_path = "traind_agents/agent_name_env_name_date"
    create_dir(save_path)

    train_report = agent.train_agent(env, main_params[train_params])

    save_train_info_to_file(train_report, save_path)

    agent.save_agent(save_path)

    val_test = find_val_from_name(main_params[val_name])
    val_report = val_test.run(agent)

    save_val_info_to_file(val_report, save_path)
    
    
    """

    # Load main config file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Verify loaded params
    main_params = config['main_params']
    print(main_params)

    agent_params = config['agent_params'][main_params['agent']]
    print(agent_params)

    env_params = config['env_params'][main_params['env']]
    print(env_params)
