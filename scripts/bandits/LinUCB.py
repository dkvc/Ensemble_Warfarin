import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

try:
    from .MultiArmedBandit import MultiArmedBandit
except ImportError:
    from MultiArmedBandit import MultiArmedBandit

class LinUCB(MultiArmedBandit):
    def __init__(self, dataset, bins, dropna=False, alpha=1.0):
        super().__init__(dataset, bins, dropna)
        self.alpha = alpha
        self.models = [make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), LinearRegression()) for _ in range(self.num_arms)]
        self.correct_predictions = np.zeros(self.num_arms)
        self.arm_counts = np.zeros(self.num_arms)
        self.A = [np.eye(dataset.shape[1] - 1) for _ in range(self.num_arms)]
        self.b = [np.zeros(dataset.shape[1] - 1) for _ in range(self.num_arms)]

    def pull_arm(self, x):
        ucb_values = []
        x = x.reshape(1, -1).astype(np.float64)
        for i in range(self.num_arms):
            if self.arm_counts[i] == 0:
                ucb_values.append(float('inf'))
            else:
                A_inv = np.linalg.inv(self.A[i])
                theta = A_inv.dot(self.b[i])
                ucb = x.dot(theta) + self.alpha * np.sqrt(x.dot(A_inv).dot(x.T))
                ucb_values.append(ucb.item())
        return np.argmax(ucb_values)

    def update(self, arm_index, x, reward):
        x = x.reshape(-1, 1)
        self.A[arm_index] += x @ x.T
        self.b[arm_index] += reward * x.flatten()
        self.arm_counts[arm_index] += 1

    def reset(self):
        self.correct_predictions = np.zeros(self.num_arms)
        self.arm_counts = np.zeros(self.num_arms)
        self.A = [np.eye(self.dataset.shape[1] - 1) for _ in range(self.num_arms)]
        self.b = [np.zeros(self.dataset.shape[1] - 1) for _ in range(self.num_arms)]

    def train(self, X_train, y_train):
        for i in range(self.num_arms):
            bin_mask = (y_train >= self.bins[i].left) & (y_train <= self.bins[i].right)
            if bin_mask.any():
                self.models[i].fit(X_train[bin_mask], y_train[bin_mask])

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def score(self, X_test=None, y_test=None):
        if not (isinstance(X_test, pd.DataFrame) and isinstance(y_test, pd.Series)):
            test_df = pd.read_csv("./data/test.csv")
            test_df.dropna(inplace=True)
            X_test = test_df.drop(columns=['Therapeutic Dose of Warfarin'])
            y_test = test_df['Therapeutic Dose of Warfarin']

        total_predictions = 0
        correct_predictions = 0
        bin_accuracy = []

        for i in range(len(X_test)):
            x = X_test.iloc[i].values.reshape(1, -1).astype(np.float64)
            arm = self.pull_arm(x)
            x = X_test.iloc[i].values.reshape(1, -1).astype(np.float64)
            prediction = self.models[arm].predict(x)
            total_predictions += 1
            if self.bins.get_indexer([y_test.iloc[i]])[0] == arm:
                correct_predictions += 1
            #self.update(arm, x.flatten(), correct_predictions)
            
            bin_accuracy.append((self.bins[arm], correct_predictions / total_predictions))

        accuracy = correct_predictions / total_predictions
        return accuracy


if __name__ == "__main__":
    train_df = pd.read_csv("./data/train.csv")
    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    X_train = train_df.drop(columns=['Therapeutic Dose of Warfarin'])
    y_train = train_df['Therapeutic Dose of Warfarin']

    model = LinUCB(train_df, bins)
    model.train(X_train, y_train)
    model.save('./models/bandits/LinUCB.pkl')

    with open('./models/bandits/LinUCB.pkl', 'rb') as file:
        model = pickle.load(file)
    accuracy = model.score()
    print(f"Accuracy: {accuracy}")
    # print("Bin Accuracy:")
    # for bin, acc in bin_accuracy:
    #     print(f"Bin: {bin}, Accuracy: {acc}")