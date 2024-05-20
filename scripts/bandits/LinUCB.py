import numpy as np
import pandas as pd
import pickle

class LinUCB():
    def __init__(self, dataset, bins, alpha=1.0):
        self.dataset = dataset
        self.bins = bins
        self.alpha = alpha
        self.num_features = self.dataset.shape[1]
        self.num_arms = len(bins)
        self.reset()

    def reset(self):
        self.A = [np.identity(self.num_features) for _ in range(self.num_arms)]
        self.b = [np.zeros(self.num_features) for _ in range(self.num_arms)]
        self.prediction = []

    def pull_arm(self, features_row):
        p = [None for _ in range(self.num_arms)]
        for arm in range(self.num_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = np.dot(A_inv, self.b[arm])
            p[arm] = np.dot(theta, features_row) + \
                     self.alpha * np.sqrt(np.dot(np.dot(features_row.T, A_inv), features_row))
        return int(np.argmax(p))

    def update(self, arm_index, features_row, reward):
        self.A[arm_index] += np.outer(features_row.T, features_row)
        self.b[arm_index] += reward * features_row

    def train(self, X_train, y_train, reward_function):
        num_rows = X_train.shape[0]
        for row in range(num_rows):
            features_row = X_train[row]
            arm_chosen = self.pull_arm(features_row)
            reward = reward_function(arm_chosen, y_train[row])
            self.update(arm_chosen, features_row, reward)
            self.prediction.append(arm_chosen)

    def score(self, X_test, y_test, reward_function):
        predictions = []
        for features_row in X_test:
            arm_chosen = self.pull_arm(features_row)
            predictions.append(arm_chosen)
        accuracy = np.mean([reward_function(predictions[i], y_test[i]) for i in range(len(y_test))])
        return accuracy
    
    def reward_function(self, prediction, row):
        return int(self.bins[prediction].left <= row <= self.bins[prediction].right)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

def normalize(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-10)

if __name__ == "__main__":
    # train features - train_features.npy & train_labels.npy
    X_train = np.load("./data/train_features.npy")
    y_train = np.load("./data/train_labels.npy")

    # test features - test_features.npy & test_labels.npy
    X_test = np.load("./data/test_features.npy")
    y_test = np.load("./data/test_labels.npy")

    # Normalize the features
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # Define the bins for the bandit
    bins = pd.IntervalIndex.from_tuples([
    (0, 20.999),
    (20.999, 49),
    (49, 20000)
    ])

    # Train the LinUCB model
    linucb = LinUCB(X_train, bins)

    # Train the model
    linucb.train(X_train, y_train, linucb.reward_function)

    # Save the model
    linucb.save("./models/bandits/LinUCB.pkl")

    # load & test the model
    with open("./models/bandits/LinUCB.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
        accuracy = loaded_model.score(X_test, y_test, loaded_model.reward_function)
        print(f"Accuracy: {accuracy}")