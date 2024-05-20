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
    def __init__(self, dataset, bins, dropna=False):
        super().__init__(dataset, bins, dropna)
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

    def train(self, X_train, y_train, epochs=10):
        time_start = time.perf_counter()
        for _ in range(epochs):
            # randomly pick indices from permutation of X_train
            indices = np.random.permutation(len(X_train))

            for i in indices:
                true_dose = y_train.iloc[i]
                arm_index = self.bins.get_indexer([true_dose])[0]

                # Pull the arm and update the model
                sampled_values = [self.pull_arm(i) for i in range(self.num_arms)]
                predicted_arm = np.argmax(sampled_values)

                # Update the model based on the reward
                reward = 1 if predicted_arm == arm_index else 0
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
        # Calculate the accuracy of the model
        # from collections import Counter
        # print(f"True Labels: {Counter(self.true_labels)}")
        # print(f"Predicted Labels: {Counter(self.predicted_labels)}")
        
        # # show success & failures only in numbers, not scientific notation
        # print(f"Successes: {self.successes.astype(int)}")
        # print(f"Failures: {self.failures.astype(int)}")
        accuracy = sum(np.array(self.true_labels) == np.array(self.predicted_labels)) / len(self.true_labels)
        return accuracy
    
    def save(self, filename):
        return super().save(filename)
    
    def time_taken(self):
        return self.time_taken
        
    
if __name__ == "__main__":
    import pandas as pd
    import pickle
    train_df, test_df = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X_train, y_train = train_df.drop(columns=['Therapeutic Dose of Warfarin']), train_df['Therapeutic Dose of Warfarin']

    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49 ,20000)
    ])
    
    model = ThompsonSamplingBandit(train_df, bins)
    #cProfile.run('model.train(X_train, y_train, epochs=20)')
    model.train(X_train, y_train, epochs=20)
    model.save('./models/bandits/Thompson.pkl')

    # try to load the model
    with open('./models/bandits/Thompson.pkl', 'rb') as file:
        model = pickle.load(file)

        from pprint import pprint
        pprint(model.score())
        pprint(f"Time taken: {model.time_taken:.3f}")