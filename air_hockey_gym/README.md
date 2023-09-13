# AirHockeyGymnasium Package
Package containing environment definition and asset files for various versions of an air hockey table RL environment using the Gymnasium API and MuJoCo physics simulator. Package structure and setup follows [Custom Environment Creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

## Setup
To setup the package on your local machine follow the below steps.
1. Download MuJoCo [mujoco-2.3.2-linux-x86_64.tar.gz](https://github.com/deepmind/mujoco/releases/download/2.3.2/mujoco-2.3.2-linux-x86_64.tar.gz)
2. Set path to MuJoCo bin as MUJOCO_PATH variable
    * `export MUJOCO_PATH=”path/to/mujoco/bin”`
3. Install Gymnasium package
    * `pip3 install gymnasium[mujoco]` 
4. Install air_hockey_gym package locally (note this is the first air_hockey_gym directory inside the repo)
    * `pip3 install -e /path/to/air_hockey_gym`