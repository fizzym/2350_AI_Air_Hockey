# Air Hockey Env Agents

Directory for holding code that defines agents for use in AirHockeyGymEnvs.

## Agents

* x_entropy_agent_v0.py
  * Implementation of the below XEntropy_nn.py using the new RL_Agent interface
  * Uses PyTorch neural network with Cross-Entropy loss to optimize neural network
* XEntropy_nn.py
  * Cross-Entropy Neural Network
  * Uses PyTorch neural network with Cross-Entropy loss to optimize neural network
* XEntropy_numpy.py
  * Cross-Entropy Numpy
  * Uses Numpy matrix to hold weights and biases and adjusts weights using Cross-Entropy

## Adding New Agent Types/Version
When adding a new agent type or version, please follow the below instructions to ensure agent interfaces properly with the end to end training file. 
### Adding New Agent Type
1. Create a new folder in `agents` directory with name “agent_name” (whatever you want to name your new agent)
2. Create a new file in `agents/agent_name` named `agent_name_vX.py`, where X is the agent version (0 to start)
3. Implement RL_Agent interface in `agent_name_vX.py` using desired training method
4. Create a new config file in `agents/agent_name` named `agent_name_vX_config.yml`
5. Add parameters to the config file in the below format:
```
init_params:
  init_1: ...
  init_2: ...
  ...
train_params:
  train_1: ...
  trains_2: ..
``` 
  - `init_params` should match the named argument's of the agent's constructor exactly
  - `train_params` should match the named arguments of the agent’s `train_agent` method exactly
6.  Add agent class name and file name to class map in `get_agent_class_and_config` function of `end_to_end_utils.py`.

### Adding New Agent Version
Assuming you want to create a new version of the agent “agent_name”, follow the below steps.
1. In the directory `agents/agent_name` create a new file `agent_name_vX.py` where X is your new version number
2. Modify code in new file as desired
3. Add comment at top of file describing changes from previous agent version
4. Create a new config file in `agents/agent_name` with the name `agent_name_vX_config.yml`
    - Follow format described in step 5 of __Adding New Agent Type__


