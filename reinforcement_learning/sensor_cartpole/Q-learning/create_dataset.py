import time
import torch
import numpy as np

import mlflow
import mlflow.pytorch

model = mlflow.pytorch.load_model("models:/sensor_cartpole/3")

import gym
env = gym.envs.make("CartPole-v0")

def select_action(model, state):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float)
        action = model(state)
        return action.argmax().item()

data = []
num_episodes = 100
for i in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action = select_action(model, obs)
        data_point = np.append(obs, action).tolist()
        data.append(data_point)
        obs, reward, done, _ = env.step(action)
np.save("../../behaviour_cloning/sensor_cartpole/sensor_dataset", data)