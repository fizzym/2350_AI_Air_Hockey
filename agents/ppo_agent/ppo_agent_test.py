from air_hockey_gym.envs import SingleMalletAlternatingEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnMaxEpisodes

env = SingleMalletAlternatingEnv(off_def_ratio=[2,1], mal1_box_def=[(-0.9, 0.4), (-0.3, -0.4)], mal1_box_off=[(-0.9, 0.4), (-0.3, -0.4)],
                                 puck_box_def=[(0.2, 0.25), (0.6, -0.25)], puck_box_off=[(-0.7, 0.4), (-0.2, -0.4)],
                                 mal2_puck_dist_range=[0.2, 0.3], mal2_vel_range=[1, 2], mal2_box_off=[(0.3, 0.4), (0.9, -0.4)])

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=20, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5000, verbose=1)

callback = CallbackList([eval_callback, callback_max_episodes])

# model = PPO(MlpPolicy, env, verbose=1, tensorboard_log="./runs/test")
model = PPO.load("trained_models/ppo_alt2")
model.set_env(env=env)

model.learn(total_timesteps=100000000, tb_log_name="", callback=callback)

model.save("trained_models/ppo_alt3")


print("DONE")

env.render_mode = "human"

obs, _ = env.reset()

for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rew, is_done, _, _ = env.step(action)
    if is_done:
        obs, _ = env.reset()
    env.render()

env.close()
