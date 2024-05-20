from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Try to handle imports based on execution context
try:
    # Module import, use relative imports
    from .Thompson import ThompsonSamplingBandit
except ImportError:
    # Direct execution, use absolute imports
    from Thompson import ThompsonSamplingBandit

class EnsembleSamplingBandit():
    def __init__(self, dataset, bins, num_models=10, dropna=False):
        self.num_models = num_models
        self.models = [ThompsonSamplingBandit(dataset.sample(frac=1 / num_models), bins, dropna) for _ in range(num_models)]
        self.true_labels = []
        self.predicted_labels = []
        self.time_taken = 0

    def reset(self):
        for model in self.models:
            model.reset()
        self.true_labels = []
        self.predicted_labels = []
        self.time_taken = 0

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def train_model(self, model, X_train, y_train, epochs):
        model.train(X_train, y_train, epochs)
        return model.true_labels, model.predicted_labels

    def train(self, X_train, y_train, epochs=10):
        epochs = epochs // self.num_models
        self.time_taken = time.perf_counter()
        with ProcessPoolExecutor(max_workers=self.num_models) as executor:
            futures = [executor.submit(self.train_model, model, X_train, y_train, epochs) for model in self.models]
            for future in as_completed(futures):
                true_labels, predicted_labels = future.result()
                self.true_labels.extend(true_labels)
                self.predicted_labels.extend(predicted_labels)    
        self.time_taken = time.perf_counter() - self.time_taken

    def score(self):
        import numpy as np
        ensemble_accuracy = sum(np.array(self.true_labels) == np.array(self.predicted_labels)) / len(self.true_labels)
        return ensemble_accuracy
    
    def time_taken(self):
        return self.time_taken

if __name__ == "__main__":
    import cProfile
    import pandas as pd
    import pickle
    train_df, test_df = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X_train, y_train = train_df.drop(columns=['Therapeutic Dose of Warfarin']), train_df['Therapeutic Dose of Warfarin']

    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    model = EnsembleSamplingBandit(train_df, bins)
    
    # cprofile across multiple pools
    #cProfile.run('model.train(X_train, y_train, epochs=20)')
    model.train(X_train, y_train, epochs=20)

    model.save('./models/bandits/Ensemble.pkl')

    # Try to load the model
    with open('./models/bandits/Ensemble.pkl', 'rb') as file:
        model = pickle.load(file)

        from pprint import pprint
        pprint(f'Ensemble Accuracy: {model.score()}')
        pprint(f"Time taken: {model.time_taken:.3f}")