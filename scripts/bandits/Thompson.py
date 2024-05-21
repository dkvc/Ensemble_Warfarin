import numpy as np
import time

# Try to handle imports based on execution context
try:
    # Module import, use relative imports
    from .MultiArmedBandit import MultiArmedBandit
except ImportError:
    # Direct execution, use absolute imports
    from MultiArmedBandit import MultiArmedBandit

class ThompsonSamplingBandit(MultiArmedBandit):
    def __init__(self, bins, dropna=False):
        self.bins = bins
        self.num_arms = len(bins)
        self.time_taken = 0
        self.reset()

    def reset(self):
        # Initialize the successes and failures for each arm
        self.successes = np.zeros(self.num_arms)
        self.failures = np.zeros(self.num_arms)
        self.true_labels = []
        self.predicted_labels = []
        self.time_taken = 0

    def pull_arm(self, arm_index):
        # Generate a sample from the Beta distribution for the given arm
        sample = np.random.beta(self.successes[arm_index] + 1, self.failures[arm_index] + 1)
        #print(f"Arm {arm_index} - Beta Sample: {sample}")
        return sample
    
    def update(self, arm_index, reward):
        # Update the successes and failures based on the reward
        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1

    def reward_function(self, true_arm, predicted_arm):
        return 1 if true_arm == predicted_arm else 0

    def train(self, X_train, y_train, epochs=10):
        time_start = time.perf_counter()
        for _ in range(epochs):
            # randomly pick indices from permutation of X_train
            indices = np.random.permutation(len(X_train))

            for i in indices:
                arm_index = y_train[i]

                # Pull the arm and update the model
                sampled_values = [self.pull_arm(i) for i in range(self.num_arms)]
                predicted_arm = np.argmax(sampled_values)

                # Update the model based on the reward
                reward = self.reward_function(arm_index, predicted_arm)
                self.update(predicted_arm, reward)

                # Debug
                # print(f"""-----------------\n
                # True Dose: {true_dose}, Arm Index: {arm_index}\n
                # Predicted Arm: {predicted_arm}, Predicted Dose: {predicted_dose}\n
                # Reward: {reward}\n
                # -----------------\n""")

                # Store the true and predicted labels for evaluation
                self.true_labels.append(arm_index)
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
        return super().save(filename)
    
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
    
    model = ThompsonSamplingBandit(bins)
    #cProfile.run('model.train(X_train, y_train, epochs=20)')
    model.train(X_train, y_train, epochs=100)
    model.save('./models/bandits/Thompson.pkl')

    # try to load the model
    with open('./models/bandits/Thompson.pkl', 'rb') as file:
        model = pickle.load(file)

        from pprint import pprint
        pprint(model.score())
        pprint(model.f1_score())
        pprint(f"Time taken: {model.time_taken:.3f}")