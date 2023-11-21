# Air Hockey Gym Agent Validation
Directory for holding files that test agent performance. Could be through unit tests (e.g. shoot puck at agent 100 times and record statistics on how often it can block the shot) or through competition with other agents.


## Adding New Validation Tests
The steps to add a new environment are as follows:
1. Create main validation test Python file in this directory with version number e.g. `test_name_vX.py`. 
2. Create config file in `/configs` directory, using name `test_name_vX_config.yml`.
    - Add name of class under `class_name` heading.
    - Add all named arguments of test's `test_agent` method under the `test_params` dictionary.
      
      (See below example)
```
---
class_name: ...

test_params:
  test_param_1: ...
  test_param_2: ...
  ...
``` 