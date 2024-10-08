import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.bandits import *

base_path = "./saved/models/"
bandits = ["bandits/thompson", "bandits/ensemble"]
loaded_models = [ThompsonSampling, EnsembleSampling]
cumulative_rewards = []
regrets = []
time_taken_models = []

def normalize(features):
    mean = torch.mean(features, axis=0)
    std = torch.std(features, axis=0)
    return (features - mean) / (std + 1e-10)

for i, bandit in enumerate(bandits):
    model_path = base_path + bandit + ".pt"
    
    X_train = np.load("./data/modified/deep_cleaned_features.npy")
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_train = normalize(X_train)
    
    y_train = np.load("./data/modified/deep_cleaned_labels.npy")
    y_train = torch.tensor(y_train, dtype=torch.float32)

    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    loaded_model = loaded_models[i](X_train, bins)
    loaded_model.load(model_path)

    cumulative_reward = loaded_model.cumulative_reward
    print(f"Cumulative Reward: {cumulative_reward}")
    cumulative_rewards.append(cumulative_reward)

    regret = loaded_model.regret
    print(f"Regret: {regret}")
    regrets.append(regret)

    print("Time taken:", loaded_model.time_taken)
    loaded_models.append(loaded_model)

num_models = []
cumulative_rewards_ensem = []
regrets_ensem = []
time_taken_ensem = []
for i in range(5, 100, 10):
    print("Number of models:", i)
    num_models.append(i)
    avg = []
    iters = 5
    for _ in range(iters):
        ensemble_sampling = EnsembleSampling(X_train, bins, num_models=i)
        ensemble_sampling.train(X_train, y_train, ensemble_sampling.reward_function)

        cumulative_reward = ensemble_sampling.cumulative_reward
        regret = ensemble_sampling.regret
        time_taken = ensemble_sampling.time_taken

        avg.append([cumulative_reward, regret, time_taken])

    cumulative_reward = sum(iter[0] for iter in avg) // len(avg)
    regret = sum(iter[1] for iter in avg) // len(avg)
    time_taken = sum(iter[2] for iter in avg) // len(avg)

    print(f"Avg Cumulative Reward: {cumulative_reward}, Avg. Regret: {regret}, Avg. Time taken: {time_taken} over {iters} iters")

    cumulative_rewards_ensem.append(cumulative_reward)
    regrets_ensem.append(regret)
    time_taken_ensem.append(time_taken)

# Plotting
plt.figure(figsize=(12, 8))

# Cumulative Reward Comparison
plt.subplot(3, 1, 1)
#plt.plot([1, 2], cumulative_rewards, 'ro-', label='Bandits')
plt.plot(num_models, cumulative_rewards_ensem, 'bo-', label='Ensemble Sampling')
plt.title('Cumulative Rewards')
plt.ylabel('Cumulative Reward')
plt.legend()

# Regret Comparison
plt.subplot(3, 1, 2)
#plt.plot([1, 2], regrets, 'ro-', label='Bandits')
plt.plot(num_models, regrets_ensem, 'bo-', label='Ensemble Sampling')
plt.title('Regrets')
plt.ylabel('Regret')
plt.legend()

# Time Taken for Models
plt.subplot(3, 1, 3)
#plt.plot([1, 2], time_taken_models, 'ro-', label='Bandits')
plt.plot(num_models, time_taken_ensem, 'bo-', label='Ensemble Sampling')
plt.title('Time taken')
plt.ylabel('Time taken')
plt.legend()

plt.show()