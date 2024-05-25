import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pandas as pd

class EnsembleModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(EnsembleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def score(self, y_true, y_pred):
        """Calculate the accuracy of the model's predictions."""
        correct_predictions = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == pred_label)
        total_predictions = len(y_true)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy

    def f1_score(self, y_true, y_pred):
        """Calculate the F1-score of the model's predictions."""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average='weighted')

class EnsembleSamplingBandit:
    def __init__(self, bins, num_models=5, input_size=10, output_size=3, learning_rate=0.01, device='cpu', reward_function=None):
        self.bins = bins
        self.num_models = num_models
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device
        self.reward_function = reward_function or self.default_reward_function  # Default if none provided

        self.models = [EnsembleModel(input_size, output_size).to(device) for _ in range(num_models)]
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]

        # Initialize success and failure counts for each arm across all models
        self.successes = torch.zeros((self.num_models, self.output_size), device=self.device)
        self.failures = torch.zeros((self.num_models, self.output_size), device=self.device)

        self.true_labels = []
        self.predicted_labels = []
        self.time_taken = 0

    def pull_arm(self, model_index, input_data):
        with torch.no_grad():
            output = self.models[model_index](torch.tensor(input_data, dtype=torch.float32).to(self.device))
            predicted_arm = torch.argmax(output).item()
        return predicted_arm

    def update(self, model_index, input_data, true_arm, reward):
        # Forward pass
        output = self.models[model_index](torch.tensor(input_data, dtype=torch.float32).to(self.device))

        # Convert true_arm to a one-hot encoded tensor (with Float type)
        true_arm_tensor = torch.zeros(self.output_size, dtype=torch.float32).to(self.device)
        true_arm_tensor[true_arm] = 1  

        # Calculate loss
        loss = nn.CrossEntropyLoss()(output, true_arm_tensor) * (1 - reward)  # Penalize incorrect predictions

        # Backward pass and optimization
        self.optimizers[model_index].zero_grad()
        loss.backward()
        self.optimizers[model_index].step()

    def default_reward_function(self, true_arm, predicted_arm):
        return 1 if true_arm == predicted_arm else 0

    def train(self, X_train, y_train, epochs=10):
        time_start = time.perf_counter()
        for _ in range(epochs):
            indices = np.random.permutation(len(X_train))

            for i in indices:
                input_data = X_train[i]
                true_arm = y_train[i]

                # Choose a model at random
                model_index = np.random.randint(self.num_models)

                # Pull an arm based on the chosen model
                predicted_arm = self.pull_arm(model_index, input_data)

                # Calculate reward based on your custom reward function
                reward = self.reward_function(true_arm, predicted_arm)

                # Update the chosen model based on the reward
                self.update(model_index, input_data, true_arm, reward)

                # Store the true and predicted labels for evaluation
                self.true_labels.append(true_arm)
                self.predicted_labels.append(predicted_arm)
        self.time_taken = time.perf_counter() - time_start

    def score(self):
        correct_predictions = sum(1 for true_label, pred_label in zip(self.true_labels, self.predicted_labels) if true_label == pred_label)
        total_predictions = len(self.true_labels)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy

    def f1_score(self):
        from sklearn.metrics import f1_score
        return f1_score(self.true_labels, self.predicted_labels, average='weighted')

    def save(self, filename):
        torch.save(self.models, filename)

    def time_taken(self):
        return self.time_taken
        
def normalize(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-10)

# Example Usage
if __name__ == "__main__":
    # Load the cleaned features and labels -- warfarin dataset
    X_train, y_train = np.load('./data/cleaned_features.npy'), np.load('./data/cleaned_labels.npy')
    X_train = normalize(X_train)

    # Define the dose bins
    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    # Initialize the ensemble sampling bandit
    input_size = X_train.shape[1]  # Get the actual input size from your data
    ensemble_bandit = EnsembleSamplingBandit(bins, num_models=5, input_size=input_size, output_size=3, learning_rate=0.01, device='cuda')

    # Train the ensemble sampling bandit
    ensemble_bandit.train(X_train, y_train, epochs=10)

    # Save the ensemble sampling bandit model
    ensemble_bandit.save('./models/bandits/EnsembleSampling.pkl')

    # Try to load the ensemble sampling bandit model
    model = torch.load('./models/bandits/EnsembleSampling.pkl')

    # Evaluate performance using a loop through each model
    accuracy = 0
    f1_score = 0
    for i in range(len(model)):
        model[i].eval()  # Set the model to eval mode
        with torch.no_grad():  # Disable gradients
            y_pred = [torch.argmax(model[i](torch.tensor(X_train[j], dtype=torch.float32).to(ensemble_bandit.device))).item() for j in range(len(X_train))]
            accuracy += model[i].score(y_train, y_pred)
            f1_score += model[i].f1_score(y_train, y_pred)

    # Average the results from all models
    accuracy /= len(model)
    f1_score /= len(model)

    # Print the results
    print(f'EnsembleSampling Accuracy: {accuracy}')
    print(f"Time taken: {ensemble_bandit.time_taken:.3f}")
    print(f"EnsembleSampling F1 Score: {f1_score}")