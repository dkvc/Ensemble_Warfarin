import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        self.reset()
        self.rewards = []

    def reset(self):
        # Initialize the successes and failures for each arm
        self.successes = np.zeros(self.num_arms)
        self.failures = np.zeros(self.num_arms)
        self.true_labels = []
        self.predicted_labels = []
        self.rewards = []

    def pull_arm(self, arm_index):
        # Generate a sample from the Beta distribution for the given arm
        sample = np.random.beta(self.successes[arm_index] + 1, self.failures[arm_index] + 1)
        print(f"Arm {arm_index} - Beta Sample: {sample}")
        return sample
    
    def plot_beta_distribution(self):
        # Plot the Beta distribution for each arm
        from scipy import stats
        import matplotlib.pyplot as plt
        x = np.linspace(0, 1, 100)
        for i in range(self.num_arms):
            y = stats.beta.pdf(x, self.successes[i] + 1, self.failures[i] + 1)
            plt.plot(x, y, label=f'Arm {i}')
        plt.legend()
        plt.show()

    def update(self, arm_index, reward):
        # Update the successes and failures based on the reward
        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1

    def train(self, X_train, y_train, epochs=1):
        #num_initial_exploration = self.num_arms * 10
        #initial_indices = np.random.choice(len(X_train), num_initial_exploration, replace=True)
        
        print(X_train)
        #p = y_train.iloc[initial_indices[0]]
        #print(p)
        print(f"Self.successes: {self.successes}, Self.failures: {self.failures}")

        #from time import sleep
        #sleep(15)

        #Initial exploration phase
        # for i in initial_indices:
        #     true_dose = y_train.iloc[i]
        #     arm_index = self.bins.get_indexer([true_dose])[0]
        #     sampled_values = [self.pull_arm(i) for i in range(self.num_arms)]
        #     predicted_arm = np.argmax(sampled_values)
        #     print(f"True dose: {true_dose}, Arm index: {arm_index}, Predicted arm: {predicted_arm}")
        #     reward = 1 if predicted_arm == arm_index else max(0, 1 - abs(predicted_arm - arm_index) / self.num_arms)
        #     self.update(predicted_arm, reward)
        #     self.true_labels.append(arm_index)
        #     self.predicted_labels.append(predicted_arm)
        for _ in range(epochs):
            # self.true_labels = []
            # self.predicted_labels = []

            indices = np.random.permutation(len(X_train))
            # in order instead of random
            #indices = np.arange(len(X_train))

            for i in indices:
                true_dose = y_train.iloc[i]
                sampled_values = [self.pull_arm(i) for i in range(self.num_arms)]
                predicted_arm = np.argmax(sampled_values)
                
                true_bin = self.bins.get_indexer([true_dose])[0]
                predicted_bin = self.bins.get_indexer([predicted_arm])[0]
                #predicted_dose = self.get_dose_from_arm(predicted_arm)
                print(f"True dose: {true_dose}, Arm index: {predicted_arm}, Predicted arm: {predicted_arm}")
                
                reward = 1 if predicted_bin == true_bin else max(0, 1 - abs(predicted_bin - true_bin) / self.num_arms)
                
                self.update(predicted_arm, reward)
                self.true_labels.append(true_dose)
                self.predicted_labels.append(predicted_arm)
                self.rewards.append(reward)

    def get_dose_from_arm(self, arm_index):
        # Convert arm index back to dose range
        if arm_index == 0:
            return self.bins[0].left + (self.bins[0].right - self.bins[0].left) / 2
        elif arm_index == 1:
            return self.bins[1].left + (self.bins[1].right - self.bins[1].left) / 2
        elif arm_index == 2:
            return self.bins[2].left + (self.bins[2].right - self.bins[2].left) / 2
        else:
            raise ValueError("Invalid arm index")

    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def score(self):
        for i in range(len(self.true_labels)):
            print(f"True dose: {self.true_labels[i]}, Predicted dose: {self.predicted_labels[i]}")

        binned_true_labels = pd.cut(self.true_labels, self.bins, labels=False)
        binned_predicted_labels = pd.cut(self.predicted_labels, self.bins, labels=False)
        # Calculate various metrics
        accuracy = accuracy_score(binned_true_labels, binned_predicted_labels)
        precision = precision_score(binned_true_labels, binned_predicted_labels, average=None, zero_division=0)
        recall = recall_score(binned_true_labels, binned_predicted_labels, average=None, zero_division=0)
        f1 = f1_score(binned_true_labels, binned_predicted_labels, average=None, zero_division=0)

        # Calculate the mean of precision, recall, and F1 score
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        mean_f1 = np.mean(f1)

        # Debug print statements to check successes and failures
        print(f"Successes: {self.successes}")
        print(f"Failures: {self.failures}")

        # The score can be defined as the expected success rate for each arm
        total = self.successes + self.failures
        success_rate = np.divide(self.successes, total, out=np.zeros_like(self.successes), where=total != 0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'mean_precision': mean_precision,
            'recall': recall,
            'mean_recall': mean_recall,
            'f1': f1,
            'mean_f1': mean_f1,
            'success_rate': success_rate,
            'average_reward': np.mean(self.rewards)
        }

        print(metrics)
        return accuracy

if __name__ == "__main__":
    # Usage example
    # Assuming train.csv and test.csv are already loaded as pandas DataFrames
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    # Define the bins for discretizing the therapeutic doses
    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    # Create an instance of the ThompsonSamplingBandit
    ts_bandit = ThompsonSamplingBandit(train_df, bins)

    # Split the dataset into features and labels for training
    X_train = train_df.drop(columns=['Therapeutic Dose of Warfarin'])
    y_train = train_df['Therapeutic Dose of Warfarin']
    print(y_train.value_counts(bins=bins))
    from time import sleep
    sleep(2)

    import cProfile

    for i in range(20):
        # Train the bandit
        #cProfile.run("ts_bandit.train(X_train, y_train)")
        ts_bandit.train(X_train, y_train)

        # Save the model
        ts_bandit.save('./models/bandits/Thompson.pkl')

        # try to load model
        with open('./models/bandits/Thompson.pkl', 'rb') as file:
            ts_bandit = pickle.load(file)
            ts_bandit.plot_beta_distribution()

            # Get the score
            score = ts_bandit.score()
            from pprint import pprint
            pprint(score)
            
            # keypress wait
            input("Press Enter to continue...")
