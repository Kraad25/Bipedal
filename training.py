from stable_baselines3 import PPO
from Environment import BipedalWalkerEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing

def make_env():
    def _init():
        return BipedalWalkerEnv(render_mode=None, MaxSteps=500)
    return _init

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Optional on Windows if not freezing an executable

    num_envs = 2  # Adjust based on CPU cores
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    check_env(BipedalWalkerEnv(MaxSteps=500))
    try:
        model = PPO.load("ppo_bipedal_standing", env=env)
        print("Loaded existing model")
    except:
        model = PPO("MlpPolicy", env)
        print("Training new model")

    model.learn(total_timesteps=2000_000, progress_bar=True)

    model.save("ppo_bipedal_standing")
    env.close()