import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Try to handle imports based on execution context
try:
    # Module import, use relative imports
    from .MultiArmedBandit import MultiArmedBandit
    from .Thompson_Old import ThompsonSamplingBandit
except ImportError:
    # Direct execution, use absolute imports
    from MultiArmedBandit import MultiArmedBandit
    from scripts.bandits.Thompson_Old import ThompsonSamplingBandit

class EnsembleSampling(MultiArmedBandit):
    def __init__(self, dataset, bins, num_models=10, dropna=False):
        super().__init__(dataset, bins, dropna)
        self.num_models = num_models
        self.models = [ThompsonSamplingBandit(dataset.sample(frac=1 / num_models), bins, dropna) for _ in range(num_models)]
        self.true_labels = []
        self.predicted_labels = []

    def reset(self):
        for model in self.models:
            model.reset()
        self.true_labels = []
        self.predicted_labels = []

    def pull_arm(self, arm_index):
        # Not needed in this implementation
        pass

    def update(self, arm_index, reward):
        # Not needed in this implementation
        pass

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def train(self, X_train, y_train, T=100):
        for t in range(T):
            for i in range(len(X_train)):
                true_dose = y_train.iloc[i]
                arm_index = self.bins.get_indexer([true_dose])[0]
                m = np.random.choice(self.num_models)
                model = self.models[m]
                sampled_values = [model.pull_arm(i) for i in range(self.num_arms)]
                predicted_arm = np.argmax(sampled_values)
                reward = 1 if predicted_arm == arm_index else 0
                model.update(predicted_arm, reward)
                self.true_labels.append(arm_index)
                self.predicted_labels.append(predicted_arm)

    def score(self):
        # Calculate scores for each model in the ensemble
        model_scores = [model.score() for model in self.models]
        print(model_scores)

        # Aggregate scores across models (e.g., take the mean)
        accuracy = np.mean([score['accuracy'] for score in model_scores])
        precision = np.mean([score['precision'] for score in model_scores], axis=0)
        recall = np.mean([score['recall'] for score in model_scores], axis=0)
        f1 = np.mean([score['f1'] for score in model_scores], axis=0)

        # Calculate mean of precision, recall, and F1 score
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        mean_f1 = np.mean(f1)

        # Aggregate success rates across models
        total_successes = np.sum([model.successes for model in self.models], axis=0)
        total_failures = np.sum([model.failures for model in self.models], axis=0)
        total = total_successes + total_failures
        success_rate = np.divide(total_successes, total, out=np.zeros_like(total_successes), where=total != 0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'mean_precision': mean_precision,
            'recall': recall,
            'mean_recall': mean_recall,
            'f1': f1,
            'mean_f1': mean_f1,
            'success_rate': success_rate
        }

        print(metrics)
        return metrics

    

# Usage example
if __name__ == "__main__":
    # Assuming train.csv and test.csv are already loaded as pandas DataFrames
    train_df = pd.read_csv('./data/cleaned.csv')
    test_df = pd.read_csv('./data/test.csv')

    # Define the bins for discretizing the therapeutic doses
    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    # Create an instance of the EnsembleSampling
    es_bandit = EnsembleSampling(train_df, bins)

    # Split the dataset into features and labels for training
    X_train = train_df.drop(columns=['Therapeutic Dose of Warfarin'])
    y_train = train_df['Therapeutic Dose of Warfarin']

    import cProfile
    for i in range(20):
        # Train the bandit
        cProfile.run("es_bandit.train(X_train, y_train, T=100)")

        # Save the model
        es_bandit.save('./models/bandits/Ensemble.pkl')

        # try to load model
        with open('./models/bandits/Ensemble.pkl', 'rb') as file:
            es_bandit = pickle.load(file)

            # Get the score
            score = es_bandit.score()
            from pprint import pprint
            #pprint(score)