from stable_baselines3 import PPO
from Environment import BipedalWalkerEnv
from stable_baselines3.common.env_checker import check_env

# Test trained agent
def evaluate_model(model, env):
    done = False
    obs, _ = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()

if __name__ == "__main__":
    mode="Walking"
    env = BipedalWalkerEnv(mode=mode)

    if mode=="Standing":
        try:
            model = PPO.load("ppo_bipedal_standing", env=env)
            print("Loaded Standing Model")
        except Exception as e:
            print("Train the model first")
            exit(1)
    if mode=="Walking":
        try:
            model = PPO.load("ppo_bipedal_walking", env=env)
            print("Loaded Standing Model")
        except Exception as e:
            print("Train the model first")
            exit(1)

    evaluate_model(model, env)