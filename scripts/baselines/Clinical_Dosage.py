import pickle
import pandas as pd

dataset = pd.read_csv('./data/cleaned.csv')
bins = pd.IntervalIndex.from_tuples([
    (0, 20.999),
    (20.999, 49),
    (49, 20000)
])

class ClinicalDosage:
    def __init__(self, dataset, bins, dropna=True):
        dataset.dropna(inplace=dropna)
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

    def predict(self, dataset):
        return dataset['Clinical Dose']

    def score(self):
        return sum(self._dataset['Clinical Dose'] == self._dataset['Therapeutic Dose of Warfarin']) / len(self._dataset)
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

# TODO: write under __main__
model = ClinicalDosage(dataset, bins)
print(f"Accuracy: {model.score()}")

model.save('./models/baselines/ClincalDosage.pkl')

# try to load model
with open('./models/baselines/ClincalDosage.pkl', 'rb') as file:
    model = pickle.load(file)
    print(f"Accuracy: {model.score()}")