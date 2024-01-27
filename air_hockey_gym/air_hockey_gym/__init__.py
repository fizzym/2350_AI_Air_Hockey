from gymnasium.envs.registration import register

register(
    id='AirHockey-v0',
    entry_point='air_hockey_gym.envs.air_hockey_v0:AirHockeyEnv',
)