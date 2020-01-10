import os
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch import optim
import random
import os

assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

import gym
env = gym.envs.make("CartPole-v0")

import mlflow
import mlflow.pytorch
mlflow.set_experiment("sensor_cartpole")

from sensor_cartpole_model import QNetwork


class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    if it <= 1000:
        return 0.5 - it * 0.00045
    return 0.05

def select_action(model, state, epsilon):
    # first determine if we go random
    if random.uniform(0, 1) <= epsilon:
        return 0 if random.uniform(0,1) <= 0.5 else 1
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float)
        action = model(state)
        return action.argmax().item()

def compute_q_val(model, state, action):
    q_vals = model(state)
    return q_vals.gather(1, action.view(-1,1))


def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    targets = reward + (torch.ones(done.shape, dtype=torch.float) - done) * discount_factor * model(next_state).max(1)[0]
    return targets.view(-1,1)

def train(model, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean
    
    # compute the q value
    q_val = compute_q_val(model, state, action)
    
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(model.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        s = env.reset()
        steps = 0
        done = False
        while not done:
            a = select_action(model, s, get_epsilon(global_steps))
            s_next, r, done, _ = env.step(a)
            memory.push((s, a, r, s_next, done))
            steps += 1
            global_steps += 1
            loss = train(model, memory, optimizer, batch_size, discount_factor)
            
            s = s_next
        mlflow.log_metric("steps", steps)
        episode_durations.append(steps)
    return episode_durations

# Let's run it!
num_episodes = 200
batch_size = 64
discount_factor = 0.8
learn_rate = 1e-3
memory = ReplayMemory(10000)
num_hidden = 128
seed = 42  # This is not randomly chosen

# We will seed the algorithm (before initializing QNetwork!) for reproducability
random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

model = QNetwork(num_hidden)
with mlflow.start_run():
    mlflow.log_param("num_episode", num_episodes)
    mlflow.log_param("discount_factor", discount_factor)
    mlflow.log_param("num_hidden", num_hidden)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learn_rate)
    mlflow.log_param("algorithm", "Q-learning")

    episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)

    # And see the results
    def smooth(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    fig = plt.figure(figsize=(8,6))
    plt.plot(smooth(episode_durations, 10))
    plt.savefig("./durations_smooth.png")
    fig.clear()
    mlflow.log_artifact("./durations_smooth.png")
    os.remove("./durations_smooth.png")

    # test 100 episodes!
    total_steps = []
    for i in range (100):
        obs = torch.from_numpy(env.reset()).type(torch.float)
        done = False
        steps = 0
        while not done:
            action = model(obs).argmax().item()
            obs, reward, done, _ = env.step(action)
            obs = torch.from_numpy(obs).type(torch.float)
            steps += 1
        total_steps.append(steps)
    mlflow.log_metric("average_duration", np.average(total_steps))

    mlflow.pytorch.log_model(model, "model")