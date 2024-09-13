import os
import pickle
import pandas as pd

class ClinicalDosage:
    def __init__(self, dataset, bins):
        self.bins = bins
        self._dataset = dataset
        self._dataset['Clinical Dose'] = 4.0376 - 0.2546*self._dataset['Age'] \
                                                + 0.0118*self._dataset['Height (cm)'] \
                                                + 0.0134*self._dataset['Weight (kg)'] \
                                                - 0.6752*self._dataset['Race_Asian'] \
                                                + 0.4060*self._dataset['Race_Black or African American'] \
                                                + 0.0443*self._dataset['Race_Unknown'] \
                                                + 1.2799*self._dataset['Enzyme_inducer'] \
                                                - 0.5695*self._dataset['Amiodarone (Cordarone)']
        self._dataset['Clinical Dose'] *= self._dataset['Clinical Dose']
        self._dataset['Clinical Dose'] = pd.cut(dataset['Clinical Dose'], bins)
        self._dataset['Therapeutic Dose of Warfarin'] = pd.cut(dataset['Therapeutic Dose of Warfarin'], bins)

    def predict(self):
        return self._dataset['Clinical Dose']
    
    def score(self):
        return sum(self._dataset['Therapeutic Dose of Warfarin'] == self.predict()) / len(self._dataset)
    
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

    model = ClinicalDosage(dataset, bins)
    print(f"Accuracy: {model.score()*100:.2f}%")

    model_base_path = './saved/models/baselines'
    os.makedirs(model_base_path, exist_ok=True)

    model_path = model_base_path + '/ClinicalDosage.pkl'
    print('Saving model to', model_path)
    model.save(model_path)

    # testing loading of the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        print(f"Accuracy: {model.score()*100:.2f}%")