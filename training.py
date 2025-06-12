from stable_baselines3 import PPO
from Environment import BipedalWalkerEnv
from stable_baselines3.common.env_checker import check_env

env = BipedalWalkerEnv(MaxSteps=500)
check_env(env)

try:
    model = PPO.load("ppo_standing", env=env)
    print("Loaded existing model")
except:
    model = PPO("MlpPolicy", env, verbose=1)
    print("Training new model")

model.learn(total_timesteps=40000, progress_bar=True)
model.save("ppo_standing")
env.close()