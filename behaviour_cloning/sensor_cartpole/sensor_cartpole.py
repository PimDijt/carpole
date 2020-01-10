import os
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CartPoleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).type(torch.float)
        self.Y = torch.from_numpy(Y).type(torch.long)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

numpy_dataset = np.load("./sensor_dataset.npy")
X = numpy_dataset[:,0:4]
Y = numpy_dataset[:,4]
split = int(X.shape[0]*0.9)

train_dataset = CartPoleDataset(X[:split], Y[:split])
val_dataset   = CartPoleDataset(X[split:], Y[split:])

batch_size = 64
epochs = 100
learn_rate = 1e-3
num_hidden = 128
seed = 42  # This is not randomly chosen

# We will seed the algorithm (before initializing QNetwork!) for reproducability
random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = QNetwork(num_hidden).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
criterion = torch.nn.CrossEntropyLoss()

with mlflow.start_run():
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("num_hidden", num_hidden)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learn_rate)
    mlflow.log_param("sensor_model_version", "3")
    mlflow.log_param("algorithm", "behaviour_cloning")

    for i in range(epochs):
        train_losses = []
        for j, batch in enumerate(train_data_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())

        val_losses = []
        for j, batch in enumerate(train_data_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)
            
            with torch.no_grad():
                y_hat = model(x)
            loss = criterion(y_hat, y)

            val_losses.append(loss.detach().item())

        mlflow.log_metric("train_loss", np.average(train_losses))
        mlflow.log_metric("val_loss", np.average(val_losses))

        print(f"Epoch: {i}, Train-Loss: {np.average(train_losses)}, Val-Loss: {np.average(val_losses)}")

    
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