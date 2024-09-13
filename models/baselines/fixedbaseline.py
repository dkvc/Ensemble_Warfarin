import os
import pickle
import pandas as pd

class FixedBaseLine:
    def __init__(self, dataset, bins):
        self.bins = bins
        self.cut = pd.cut(dataset['Therapeutic Dose of Warfarin'], bins)

    def predict(self):
        return self.bins[1]
    
    def score(self):
        return sum(self.cut == self.predict()) / len(self.cut)
    
    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

if __name__ == "__main__":
    dataset = pd.read_csv('./data/modified/simple_cleaned.csv')
    dataset.dropna(inplace=True)

    """
    Bins:
    < 21 mg/week
    21-49 mg/week
    > 49 mg/week
    """
    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    model = FixedBaseLine(dataset, bins)
    print(f"Accuracy: {model.score()*100:.2f}%")

    model_base_path = './saved/models/baselines'
    os.makedirs(model_base_path, exist_ok=True)
    
    model_path = model_base_path + '/FixedBaseLine.pkl'
    print('Saving model to', model_path)
    model.save(model_path)

    # testing loading of the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        print(f"Accuracy: {model.score()*100:.2f}%")