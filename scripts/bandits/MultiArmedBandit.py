from abc import ABC, abstractmethod
import pickle

class MultiArmedBandit(ABC):
    @abstractmethod
    def __init__(self, dataset, bins, dropna=False):
        if dropna:
            dataset.dropna(inplace=True)
        self.dataset = dataset.drop(columns=['Therapeutic Dose of Warfarin'])
        self.labels = dataset['Therapeutic Dose of Warfarin']
        self.bins = bins
        self.num_arms = len(bins)

    @abstractmethod
    def pull_arm(self, arm_index):
        ...

    @abstractmethod
    def update(self, arm_index, reward):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def train(self, X_train, y_train):
        ...

    @abstractmethod
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @abstractmethod
    def score(self):
        ...