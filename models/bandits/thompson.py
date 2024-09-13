import os
import time

import numpy as np
import pandas as pd
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class ThompsonSampling:
    def __init__(self, X_train, bins=None, device='cuda'):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = None
        self.bins = bins
        self.num_features = self.X_train.shape[1]
        self.num_arms = len(bins)
        self.reset()

    def reset(self):
        # Initialize mean and covariance matrices for each arm
        self.means = [torch.zeros(self.num_features, device=self.device) for _ in range(self.num_arms)]
        self.covariances = [torch.eye(self.num_features, device=self.device) for _ in range(self.num_arms)]
        
        self.counts = [0 for _ in range(self.num_arms)]
        self.prediction = []
        self.cumulative_reward = 0
        self.regret = 0

    def sample_theta(self, arm_index):
        # Sample from the posterior distribution of the mean for the chosen arm
        mean = self.means[arm_index]
        covariance = self.covariances[arm_index]
        distribution = MultivariateNormal(mean, covariance)
        return distribution.sample()

    def pull_arm(self, features_row):
        sampled_means = [torch.matmul(self.sample_theta(arm_index), features_row) for arm_index in range(self.num_arms)]
        return int(torch.argmax(torch.tensor(sampled_means)))

    def update(self, arm_index, features_row, reward):
        self.counts[arm_index] += 1
        features_row = features_row.unsqueeze(0)
        
        # Update the covariance matrix and mean for the chosen arm
        self.covariances[arm_index] += torch.matmul(features_row.T, features_row)
        self.means[arm_index] += reward * torch.matmul(torch.linalg.inv(self.covariances[arm_index]), features_row.T).squeeze()

    def train(self, X_train, y_train, reward_function):
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        num_rows = self.X_train.shape[0]

        time_start = time.perf_counter()
        for row in range(num_rows):
            features_row = self.X_train[row]
            arm_chosen = self.pull_arm(features_row)
            reward = reward_function(self.y_train[row], arm_chosen)
            self.update(arm_chosen, features_row, reward)
            self.prediction.append(arm_chosen)
            self.cumulative_reward += reward

        self.compute_regret()
        self.time_taken = time.perf_counter() - time_start

    def reward_function(self, true_arm, predicted_arm):
        if predicted_arm == true_arm:
            return 1
        else:
            return 0
    
    def optimal_policy(self, features_row):
        """
        Define the optimal policy based on y_train. 
        The optimal arm for each feature vector is based on the majority vote or model
        derived from y_train.
        """
        # Compute similarity between features_row and all rows in X_train
        similarities = torch.cdist(features_row.unsqueeze(0), self.X_train).squeeze(0)
        closest_index = torch.argmin(similarities)
        optimal_arm = int(self.y_train[closest_index].item())
        return optimal_arm
    
    def compute_regret(self):
        """
        Computes the regret of the agent. Regret is defined as the difference between the 
        cumulative reward obtained by an optimal policy and the cumulative reward obtained by the agent.
        """
        optimal_reward = 0
        for row in range(len(self.y_train)):
            features_row = self.X_train[row]
            optimal_arm = self.optimal_policy(features_row)
            optimal_reward += self.reward_function(self.y_train[row], optimal_arm)

        regret = optimal_reward - self.cumulative_reward
        self.regret = regret
        return regret

    def save(self, filepath):
        torch.save({
            'means': self.means,
            'covariances': self.covariances,
            'counts': self.counts,
            'cumulative_reward': self.cumulative_reward,
            'bins': self.bins,
            'predictions': self.prediction,
            'num_features': self.num_features,
            'num_arms': self.num_arms,
            'regret' : self.regret,
            'time_taken': self.time_taken
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.means = checkpoint['means']
        self.covariances = checkpoint['covariances']
        self.counts = checkpoint['counts']
        self.cumulative_reward = checkpoint['cumulative_reward']
        self.bins = checkpoint['bins']
        self.prediction = checkpoint['predictions']
        self.num_features = checkpoint['num_features']
        self.num_arms = checkpoint['num_arms']
        self.regret = checkpoint['regret']
        self.time_taken = checkpoint['time_taken']

def normalize(features):
    mean = torch.mean(features, axis=0)
    std = torch.std(features, axis=0)
    return (features - mean) / (std + 1e-10)

if __name__ == "__main__":
    X_train = np.load("./data/modified/deep_cleaned_features.npy")
    y_train = np.load("./data/modified/deep_cleaned_labels.npy")

    # normalize features using torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_train = normalize(X_train)

    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    thompson_sampling = ThompsonSampling(X_train, bins)
    thompson_sampling.train(X_train, y_train, thompson_sampling.reward_function)
    
    model_base_path = './saved/models/bandits'
    os.makedirs(model_base_path, exist_ok=True)

    model_path = model_base_path + "/thompson.pt"
    thompson_sampling.save(model_path)

    loaded_model = ThompsonSampling(X_train, bins)
    loaded_model.load(model_path)
    cumulative_reward = loaded_model.cumulative_reward
    print(f"Cumulative Reward: {cumulative_reward}")

    regret = loaded_model.regret
    print(f"Regret: {regret}")

    print("Time taken:", loaded_model.time_taken)