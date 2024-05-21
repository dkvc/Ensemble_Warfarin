from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np

# Try to handle imports based on execution context
try:
    # Module import, use relative imports
    from .Thompson import ThompsonSamplingBandit
except ImportError:
    # Direct execution, use absolute imports
    from Thompson import ThompsonSamplingBandit

class EnsembleSamplingBandit():
    def __init__(self, bins, num_models=10, dropna=False):
        self.num_models = num_models
        self.bins = bins
        self.true_labels = []
        self.predicted_labels = []
        self.time_taken = 0
        self.models = None

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
        # Randomly shuffle the training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]
        #print(f"Training model with {len(X_train_shuffled)} samples")
        #print(f"Training model with {len(y_train_shuffled)} samples")
        model.train(X_train_shuffled, y_train_shuffled, epochs)
        return model.true_labels, model.predicted_labels

    def train(self, X_train, y_train, epochs=10):
        epochs = epochs // self.num_models
        self.models = [ThompsonSamplingBandit(self.bins) for _ in range(self.num_models)]
        time_start = time.perf_counter()
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.train_model, model, X_train, y_train, epochs) for model in self.models]
            for future in as_completed(futures):
                #print(f"Training model {future}")
                true_labels, predicted_labels = future.result()
                self.true_labels.extend(true_labels)
                self.predicted_labels.extend(predicted_labels)
        self.time_taken = time.perf_counter() - time_start

    def score(self):
        correct_predictions = sum(1 for true_label, pred_label in zip(self.true_labels, self.predicted_labels) if true_label == pred_label)
        total_predictions = len(self.true_labels)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy
    
    def f1_score(self):
        from sklearn.metrics import f1_score
        return f1_score(self.true_labels, self.predicted_labels, average='weighted')
    
    def time_taken(self):
        return self.time_taken