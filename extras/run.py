"""
Extra scripts are run separately from main function.
It can also involve optimization of calls or time calculation for running the program.

The below script is used for results.
"""

import numpy as np
import os
import pandas as pd
import sys
import time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.bandits import *

def normalize(features):
    mean = torch.mean(features, axis=0)
    std = torch.std(features, axis=0)
    return (features - mean) / (std + 1e-10)

if __name__ == "__main__":
    X_train = np.load("../data/modified/deep_cleaned_features.npy")
    y_train = np.load("../data/modified/deep_cleaned_labels.npy")

    # Normalize features using torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_train = normalize(X_train)

    print(f"Data size: {y_train.shape}")

    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    # Create and train Ensemble Sampling Bandit model
    ensemble_train_times = []
    ensemble_cum_rewards = []
    ensemble_regrets = []
    thompson_train_times = []
    thompson_cum_rewards = []
    thompson_regrets = []
    for i in range(10):
        ensemble_sampling = EnsembleSampling(X_train, bins, num_models=5)
        ensemble_sampling.train(X_train, y_train, ensemble_sampling.reward_function)
        train_time_taken = ensemble_sampling.time_taken
        
        ensemble_train_times.append(train_time_taken)
        ensemble_cum_rewards.append(ensemble_sampling.cumulative_reward)
        ensemble_regrets.append(ensemble_sampling.regret)

        thompson_sampling = ThompsonSampling(X_train, bins)
        thompson_sampling.train(X_train, y_train, thompson_sampling.reward_function)
        t_train_time_taken = thompson_sampling.time_taken
        
        thompson_train_times.append(t_train_time_taken)
        thompson_cum_rewards.append(thompson_sampling.cumulative_reward)
        thompson_regrets.append(thompson_sampling.regret)

        test_features = X_train[0]
        test_label = y_train[0]
        t_predict_start = time.perf_counter_ns()
        predicted_arm = thompson_sampling.predict(test_features)
        t_predict = time.perf_counter_ns() - t_predict_start
        print(f"Test Label: {test_label.item()}, Predicted Arm: {predicted_arm} <-> Time: {t_predict:.2f} nanosec")

        e_predict_start = time.perf_counter_ns()
        predicted_arm2 = ensemble_sampling.predict(test_features)
        e_predict = time.perf_counter_ns() - e_predict_start
        print(f"Test Label: {test_label.item()}, Predicted Arm: {predicted_arm2} <-> Time: {e_predict:.2f} nanosec")
        
        print(f"Run {i+1}:")
        print(f"  [Thompson] Training Time: {t_train_time_taken:.2f} seconds | Cumulative Reward: {thompson_sampling.cumulative_reward} | Regret: {thompson_sampling.regret}")
        print(f"  [Ensemble] Training Time: {train_time_taken:.2f} seconds | Cumulative Reward: {ensemble_sampling.cumulative_reward} | Regret: {ensemble_sampling.regret}")

    print(f"[Thompson] Avg Time taken for Training: {sum(thompson_train_times) / len(thompson_train_times)} seconds")
    print(f"[Thompson] Avg Cumulative Reward: {sum(thompson_cum_rewards) / len(thompson_cum_rewards)}")
    print(f"[Thompson] Avg Regret: {sum(thompson_regrets) / len(thompson_regrets)}")
    print(f"[Ensemble] Avg Time taken for Training: {sum(ensemble_train_times) / len(ensemble_train_times)} seconds")
    print(f"[Ensemble] Avg Cumulative Reward: {sum(ensemble_cum_rewards) / len(ensemble_cum_rewards)}")
    print(f"[Ensemble] Avg Regret: {sum(ensemble_regrets) / len(ensemble_regrets)}")