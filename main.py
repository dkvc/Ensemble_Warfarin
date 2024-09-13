import pickle
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

from models.baselines import *
from models.bandits import *

base_path = "./saved/models/"
baselines = ["baselines/FixedBaseLine", "baselines/ClinicalDosage", "baselines/PharmogenicDosage"]
bandits = ["bandits/linucb", "bandits/thompson", "bandits/ensemble"]
loaded_models = [LinUCB, ThompsonSampling, EnsembleSampling]

accuracies = []
for baseline in baselines:
    model_path = base_path + baseline + ".pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        accuracy = model.score()
        
        print(f"Accuracy: {accuracy*100:.2f}%")
        accuracies.append(accuracy)

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