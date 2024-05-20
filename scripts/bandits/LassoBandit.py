import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

# Try to handle imports based on execution context
try:
    # Module import, use relative imports
    from .MultiArmedBandit import MultiArmedBandit
except ImportError:
    # Direct execution, use absolute imports
    from MultiArmedBandit import MultiArmedBandit

class LassoBandit(MultiArmedBandit):
    def __init__(self, dataset, bins, dropna=True, alpha=1.0):
        super().__init__(dataset, bins, dropna)
        self.alpha = alpha
        self.models = [make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), Lasso(alpha=self.alpha)) for _ in range(self.num_arms)]
        self.correct_predictions = np.zeros(self.num_arms)
        self.arm_counts = np.zeros(self.num_arms)

    def pull_arm(self):
        epsilon = 0.1
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_arms)
        else:
            return np.argmax(self.correct_predictions / (self.arm_counts + 1e-5))

    def update(self, arm_index, correct):
        self.correct_predictions[arm_index] += correct
        self.arm_counts[arm_index] += 1

    def reset(self):
        self.correct_predictions = np.zeros(self.num_arms)
        self.arm_counts = np.zeros(self.num_arms)

    def train(self, X_train, y_train):
        for i in range(self.num_arms):
            bin_mask = (y_train >= self.bins[i].left) & (y_train <= self.bins[i].right)
            if bin_mask.any():
                self.models[i].fit(X_train[bin_mask], y_train[bin_mask])

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def score(self, X_test=None, y_test=None):
        if not (isinstance(X_test, pd.DataFrame) or isinstance(y_test, pd.Series)):
            test_df = pd.read_csv("data/test.csv")
            test_df.dropna(inplace=True)
            X_test = test_df.drop(columns=['Therapeutic Dose of Warfarin'])
            y_test = test_df['Therapeutic Dose of Warfarin']

        total_predictions = 0
        correct_predictions = 0
        
        for i in range(len(X_test)):
            arm = self.pull_arm()
            X = X_test.iloc[i].values.reshape(1, -1)
            prediction = self.models[arm].predict(X)
            
            actual_value = y_test.iloc[i]
            correct = (self.bins[arm].left <= actual_value <= self.bins[arm].right)
            
            correct_predictions += int(correct)
            total_predictions += 1
            
            self.update(arm, int(correct))
        
        accuracy = correct_predictions / total_predictions
        return accuracy

if __name__ == "__main__":
    import pandas as pd
    import pickle

    # Load dataset
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")
    test_df.dropna(inplace=True)

    X_train = train_df.drop(columns=['Therapeutic Dose of Warfarin'])
    y_train = train_df['Therapeutic Dose of Warfarin']
    X_test = test_df.drop(columns=['Therapeutic Dose of Warfarin'])
    y_test = test_df['Therapeutic Dose of Warfarin']

    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    model = LassoBandit(train_df, bins)
    model.train(X_train, y_train)

    # Evaluate model on test data
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    model.save("./models/bandits/Lasso.pkl")
    
    with open("./models/bandits/Lasso.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
        print(f"Loaded model accuracy: {loaded_model.score(X_test, y_test)}")
