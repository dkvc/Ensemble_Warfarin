import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pandas as pd

class ThompsonSamplingBandit:
    def __init__(self, bins, learning_rate=0.01, device='cpu', reward_function=None):
        self.bins = bins
        self.num_arms = len(bins)
        self.device = device
        self.learning_rate = learning_rate
        self.reward_function = reward_function or self.default_reward_function
        self.reset()

    def reset(self):
        # Initialize the successes and failures for each arm as tensors
        self.successes = torch.zeros(self.num_arms, device=self.device)
        self.failures = torch.zeros(self.num_arms, device=self.device)
        self.true_labels = []
        self.predicted_labels = []
        self.time_taken = 0

        # Initialize a linear model
        self.model = nn.Linear(35, self.num_arms).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def pull_arm(self, input_data):
        # Use the linear model to predict arm probabilities
        with torch.no_grad():
            output = self.model(torch.tensor(input_data, dtype=torch.float32).to(self.device))
            # Sample an arm from the predicted probabilities using a categorical distribution
            predicted_arm = torch.distributions.Categorical(probs=torch.nn.functional.softmax(output, dim=-1)).sample().item()  # Change dim to -1
        return predicted_arm

    def update(self, arm_index, reward, input_data):
        # Update the successes and failures based on the reward
        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1

        # Train the linear model
        self.optimizer.zero_grad()
        output = self.model(torch.tensor(input_data, dtype=torch.float32).to(self.device))
        
        # Convert arm_index to a one-hot encoded tensor
        true_arm_tensor = torch.zeros(self.num_arms, dtype=torch.float32).to(self.device)  # Change dtype to torch.float32
        true_arm_tensor[arm_index] = 1 

        loss = nn.CrossEntropyLoss()(output, true_arm_tensor) * (1 - reward) 
        loss.backward()
        self.optimizer.step()

    def reward_function(self, true_arm, predicted_arm):
        return 1 if true_arm == predicted_arm else 0

    def train(self, X_train, y_train, epochs=10):
        time_start = time.perf_counter()
        for _ in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(len(X_train))

            for i in indices:
                arm_index = y_train[i]
                input_data = X_train[i]

                # Pull the arm and update the model
                predicted_arm = self.pull_arm(input_data)

                # Update the model based on the reward
                reward = self.reward_function(arm_index, predicted_arm)
                self.update(arm_index, reward, input_data)

                # Store the true and predicted labels for evaluation
                self.true_labels.append(arm_index)
                self.predicted_labels.append(predicted_arm)
        self.time_taken = time.perf_counter() - time_start

    def default_reward_function(self, true_arm, predicted_arm):
        return 1 if true_arm == predicted_arm else 0

    def score(self):
        correct_predictions = sum(1 for true_label, pred_label in zip(self.true_labels, self.predicted_labels) if true_label == pred_label)
        total_predictions = len(self.true_labels)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy
    
    def f1_score(self):
        from sklearn.metrics import f1_score
        return f1_score(self.true_labels, self.predicted_labels, average='weighted')

    def save(self, filename):
        torch.save({'true_labels': self.true_labels, 'predicted_labels': self.predicted_labels, 'model_state_dict': self.model.state_dict(), 'time_taken': self.time_taken}, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.true_labels = checkpoint['true_labels']
        self.predicted_labels = checkpoint['predicted_labels']
        self.time_taken = checkpoint['time_taken']
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def time_taken(self):
        return self.time_taken    

def normalize(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-10)

if __name__ == "__main__":
    import pandas as pd
    import pickle
    import numpy as np
    X_train, y_train = np.load('./data/cleaned_features.npy'), np.load('./data/cleaned_labels.npy')

    X_train = normalize(X_train)

    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49 ,20000)
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ThompsonSamplingBandit(bins, device=device)
    #cProfile.run('model.train(X_train, y_train, epochs=20)')
    model.train(X_train, y_train, epochs=20)
    model.save('./models/bandits/Thompson.pkl')

    # try to load the model
    model.load('./models/bandits/Thompson.pkl')

    from pprint import pprint
    pprint(model.score())
    pprint(model.f1_score())
    pprint(f"Time taken: {model.time_taken:.3f}")