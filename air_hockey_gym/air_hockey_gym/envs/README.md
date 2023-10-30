# Air Hockey Environments

Directory for holding code that defines various air hockey RL training environments. 

## Adding New Environments
The steps to add a new environment are as follows:
1. Create main environment Python file in this directory with version number e.g. `env_name_vX.py`. 
2. Create config file in `/configs` directory, using name `env_name_vX_config.yml`.
    - Add name of class under `class_name` heading.
    - Add all named arguments of environment constructor under the `init_params` dictionary.
      
      (See below example)
```
---
class_name: ...

init_params:
  init_1: ...
  init_2: ...
  ...
``` 