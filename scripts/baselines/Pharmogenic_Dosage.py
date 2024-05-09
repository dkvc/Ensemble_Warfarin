import pickle
import pandas as pd

class PharmogenicDosage:
    def __init__(self, dataset, bins, dropna=True):
        dataset.dropna(inplace=dropna) # there are no null values to drop in this case
        self._dataset = dataset
        self._dataset['Pharmogenic Dose'] = 5.6044 - 0.2614*self._dataset['Age'] \
                                                + 0.0087*self._dataset['Height (cm)'] \
                                                + 0.0128*self._dataset['Weight (kg)'] \
                                                - 0.8677*self._dataset['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T_A/G'] \
                                                - 1.6974*self._dataset['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T_A/A'] \
                                                - 0.4854*self._dataset['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T_Unknown'] \
                                                - 0.5211*self._dataset['Cyp2C9 genotypes_*1/*2'] \
                                                - 0.9357*self._dataset['Cyp2C9 genotypes_*1/*3'] \
                                                - 1.0616*self._dataset['Cyp2C9 genotypes_*2/*2'] \
                                                - 1.9206*self._dataset['Cyp2C9 genotypes_*2/*3'] \
                                                - 2.3312*self._dataset['Cyp2C9 genotypes_*3/*3'] \
                                                - 0.2188*self._dataset['Cyp2C9 genotypes_Unknown'] \
                                                - 0.1092*self._dataset['Race_Asian'] \
                                                - 0.2760*self._dataset['Race_Black or African American'] \
                                                - 0.1032*self._dataset['Race_Unknown'] \
                                                + 1.1816*self._dataset['Enzyme_inducer'] \
                                                - 0.5503*self._dataset['Amiodarone (Cordarone)']
        self._dataset['Pharmogenic Dose'] *= self._dataset['Pharmogenic Dose']
        self._dataset['Pharmogenic Dose'] = pd.cut(dataset['Pharmogenic Dose'], bins)
        self._dataset['Therapeutic Dose of Warfarin'] = pd.cut(dataset['Therapeutic Dose of Warfarin'], bins)

    def predict(self, dataset):
        return dataset['Pharmogenic Dose']

    def score(self):
        return sum(self._dataset['Pharmogenic Dose'] == self._dataset['Therapeutic Dose of Warfarin']) / len(self._dataset)
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

if __name__ == "__main__":
    dataset = pd.read_csv('./data/warfarin.csv')
    dataset = dataset.fillna('Unknown')
    dataset = dataset.loc[dataset['Age'] != 'Unknown']
    dataset = dataset.loc[dataset['Height (cm)'] != 'Unknown']
    dataset = dataset.loc[dataset['Weight (kg)'] != 'Unknown']
    dataset = dataset.loc[dataset['Carbamazepine (Tegretol)'] != 'Unknown']
    dataset = dataset.loc[dataset['Phenytoin (Dilantin)'] != 'Unknown']
    dataset = dataset.loc[dataset['Rifampin or Rifampicin'] != 'Unknown']
    dataset = dataset.loc[dataset['Amiodarone (Cordarone)'] != 'Unknown']

    # Columns required
    # Age, Height, Weight, Amiodarone, Carbamazepine (Tegretol), Phenytoin (Dilantin), Rifampin or Rifampicin
    # CYP2C9, rs9923231 VKORC1 (-1639), Race

    # Enzyme_inducer
    dataset['Carbamazepine (Tegretol)'] = dataset['Carbamazepine (Tegretol)'].map({1.0: True, 0.0: False})
    dataset['Phenytoin (Dilantin)'] = dataset['Phenytoin (Dilantin)'].map({1.0: True, 0.0:False})
    dataset['Rifampin or Rifampicin'] = dataset['Rifampin or Rifampicin'].map({1.0: True, 0.0:False})
    dataset['Enzyme_inducer'] = dataset.apply(lambda row: 1 if (row['Carbamazepine (Tegretol)'] == 1 or row['Phenytoin (Dilantin)'] == 1 or row['Rifampin or Rifampicin'] == 1) else 0, axis=1)

    # CYP2C9, rs9923231 VKORC1 (-1639), Race
    # fill NA as Unknown for these columns
    cols = ['Cyp2C9 genotypes', 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'Race']
    dataset = pd.get_dummies(dataset, prefix = cols, columns=cols)

    # map ages
    dataset['Age'] = dataset['Age'].map({'10 - 19': 1, '20 - 29': 2, '30 - 39': 3, '40 - 49': 4, '50 - 59': 5, '60 - 69': 6, '70 - 79': 7, '80 - 89' : 8, '90+' : 9})

    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    model = PharmogenicDosage(dataset, bins)
    print(f"Accuracy: {model.score()}")

    model.save('./models/baselines/PharmogenicDosage.pkl')

    # try to load model
    with open('./models/baselines/PharmogenicDosage.pkl', 'rb') as file:
        model = pickle.load(file)
        print(f"Accuracy: {model.score()}")