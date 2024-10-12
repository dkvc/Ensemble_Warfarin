import os
import time
import numpy as np
import pandas as pd
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

# temporary
import warnings
warnings.filterwarnings('ignore')

class EnsembleSampling:
    def __init__(self, X_train, bins, num_models=5, device='cuda'):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Normalize features and move to device
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = None
        self.bins = bins
        self.num_models = num_models
        self.num_features = self.X_train.shape[1]
        self.num_arms = len(bins)
        self.models = []
        self.reset()
        
    def reset(self):
        # Initialize models with mean and covariance matrices
        for _ in range(self.num_models):
            mean = torch.zeros(self.num_features, device=self.device)
            covariance = torch.eye(self.num_features, device=self.device)
            self.models.append({'mean': mean, 'covariance': covariance})
        
        self.counts = [0] * self.num_arms
        self.prediction = []
        self.cumulative_reward = 0
        self.regret = 0

    def sample_theta(self, arm_index):
        # Sample from the posterior distribution of the mean for the chosen arm
        mean = self.models[arm_index]['mean']
        covariance = self.models[arm_index]['covariance']
        distribution = MultivariateNormal(mean, covariance)
        return distribution.sample()

    def sample(self, features_row):
        sampled_means = torch.zeros(self.num_arms, device=self.device)
        for arm_index in range(self.num_arms):
            sampled_theta = self.sample_theta(arm_index)
            sampled_means[arm_index] = torch.matmul(features_row, sampled_theta).mean()
        return sampled_means

    def pull_arm(self, features_row):
        # Sample means and select arm with maximum mean
        features_row = features_row.to(self.device)
        sampled_means = self.sample(features_row)
        chosen_arm = int(torch.argmax(sampled_means))
        return chosen_arm

    def update(self, arm_index, features_row, reward):
        if arm_index < 0 or arm_index >= self.num_arms:
            raise IndexError(f"Arm index {arm_index} is out of range. Valid range is [0, {self.num_arms - 1}]")
        
        self.counts[arm_index] += 1
        features_row = features_row.unsqueeze(0)

        # Update covariance matrix and mean for the chosen arm
        self.models[arm_index]['covariance'] += torch.matmul(features_row.T, features_row)
        self.models[arm_index]['mean'] += reward * torch.matmul(torch.linalg.inv(self.models[arm_index]['covariance']), features_row.T).squeeze()

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
        return 1 if predicted_arm == true_arm else 0

    def optimal_policy(self, features_row):
        similarities = torch.cdist(features_row.unsqueeze(0), self.X_train).squeeze(0)
        closest_index = torch.argmin(similarities)
        optimal_arm = int(self.y_train[closest_index].item())
        return optimal_arm

    def compute_regret(self):
        optimal_reward = 0
        for row in range(len(self.y_train)):
            features_row = self.X_train[row]
            optimal_arm = self.optimal_policy(features_row)
            optimal_reward += self.reward_function(self.y_train[row], optimal_arm)

        regret = optimal_reward - self.cumulative_reward
        self.regret = regret
        return regret
    
    def predict(self, features_row):
        features_row = features_row.to(self.device)
        sampled_means = self.sample(features_row)
        chosen_arm = int(torch.argmax(sampled_means))
        return chosen_arm

    def save(self, filepath):
        torch.save({
            'models': self.models,
            'num_models': self.num_models,
            'counts': self.counts,
            'cumulative_reward': self.cumulative_reward,
            'bins': self.bins,
            'predictions': self.prediction,
            'num_features': self.num_features,
            'num_arms': self.num_arms,
            'regret': self.regret,
            'time_taken': self.time_taken
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.models = checkpoint['models']
        self.num_models = checkpoint['num_models']
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
    train_times = []
    for i in range(10):
        train_time_start = time.perf_counter()

        ensemble_sampling = EnsembleSampling(X_train, bins, num_models=5)
        ensemble_sampling.train(X_train, y_train, ensemble_sampling.reward_function)

        train_time_taken = time.perf_counter() - train_time_start
        train_times.append(train_time_taken)

        # Clear cache
        torch.cuda.empty_cache()

        print(f"Run {i+1}:")
        print(f"  Training Time: {train_time_taken:.2f} seconds")

    print(f"Avg Time taken for Training: {sum(train_times) / len(train_times)} seconds")
    
    model_base_path = './saved/models/bandits'
    os.makedirs(model_base_path, exist_ok=True)

    model_path = model_base_path + "/ensemble.pt"
    ensemble_sampling.save(model_path)

    loaded_model = EnsembleSampling(X_train, bins)
    loaded_model.load(model_path)
    cumulative_reward = loaded_model.cumulative_reward
    print(f"Cumulative Reward: {cumulative_reward}")

    regret = loaded_model.regret
    print(f"Regret: {regret}")

    test_features = X_train[0]
    test_label = y_train[0]
    predicted_arm = loaded_model.predict(test_features)

    print(f"Test Label: {test_label.item()}, Predicted Arm: {predicted_arm}")

    print("Time taken:", loaded_model.time_taken)