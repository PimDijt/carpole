import time
import torch

import mlflow
import mlflow.pytorch

# model = mlflow.pytorch.load_model("models:/sensor_cartpole/1")
model = mlflow.pytorch.load_model("runs:/b227c762819e4eafa51a98f88a628c6c/model")

import gym
env = gym.envs.make("CartPole-v0")

def select_action(model, state):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float)
        action = model(state)
        return action.argmax().item()

# The nice thing about the CARTPOLE is that it has very nice rendering functionality (if you are on a local environment). Let's have a look at an episode
obs = env.reset()
env.render()
done = False
while not done:
    action = select_action(model, obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.05)
env.close()  # Close the environment or you will have a lot of render screens soon