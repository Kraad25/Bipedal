import pygame
from stable_baselines3 import PPO
from Environment import BipedalWalkerEnv
from stable_baselines3.common.env_checker import check_env


# Environment
env = BipedalWalkerEnv()

# Load Models
try:
    model_stand = PPO.load("ppo_bipedal_standing", env=env)
    model_walk = PPO.load("ppo_bipedal_walking", env=env)
except Exception as e:
    print("Train the models first")
    exit(1)

obs, _ = env.reset()
done = False
mode = "Standing"

env.render()

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                mode = "Walking"
                env.mode = mode
            if event.key == pygame.K_x:
                mode = "Standing"
                env.mode = mode

    if mode == "Standing":
        action, _ = model_stand.predict(obs)
    if mode == "Walking":
        action, _ = model_walk.predict(obs)

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    env.render()

env.close()