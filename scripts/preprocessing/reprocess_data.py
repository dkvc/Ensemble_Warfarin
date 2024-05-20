import numpy as np
import csv
import pandas as pd

bins = pd.IntervalIndex.from_tuples([
    (0, 20.999),
    (20.999, 49),
    (49, 20000)
])

def load_data(file_name):
    data = []
    labels = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        _ = next(reader)  # Skip the header
        for row in reader:
            if is_complete(row):
                data.append(row)
                
                # row[4] is the 'Therapeutic Dose of Warfarin' column
                # get_bin
                labels.append(get_bin(float(row[4]), bins))

    return data, labels

def get_bin(dose, bins):
    return bins.get_indexer([dose])[0]

def is_complete(row):
    return all(row)

def all_feature_extractor(row):
    feature = []

    # Age in decades
    feature.append(float(row[0]))

    # Height in cm
    feature.append(float(row[1]))

    # Weight in kg
    feature.append(float(row[2]))

    # Amiodarone status
    feature.append(float(row[3] == '1'))

    # Enzyme inducer status
    feature.append(float(row[5] == '1'))

    # Race
    if row[6] == 'True':
        feature += [1, 0, 0, 0]
    elif row[7] == 'True':
        feature += [0, 1, 0, 0]
    elif row[8] == 'True':
        feature += [0, 0, 1, 0]
    else:
        feature += [0, 0, 0, 1]

    # CYP2C9 genotypes
    # [10:21] is the range of CYP2C9 genotypes
    genotypes = []
    for i in range(10, 21):
        if row[i] == True:
            genotypes.append(1)
        else:
            genotypes.append(0)

    feature += genotypes

    # VKORC1 genotypes
    # [21:35] is the range of VKORC1 genotypes
    genotypes = []
    for i in range(21, 35):
        if row[i] == True:
            genotypes.append(1)
        else:
            genotypes.append(0)

    feature += genotypes

    # Bias
    feature.append(1)

    return np.array(feature)

def extract_all_features(data):
    return np.array([all_feature_extractor(row) for row in data])

if __name__ == "__main__":
    names = ["train", "test"]
    for name in names:
        data, labels = load_data(f"./data/{name}.csv")
        all_features = extract_all_features(data)
        print(all_features)
        print(labels)

        np.save(f"./data/{name}_features.npy", all_features)
        np.save(f"./data/{name}_labels.npy", labels)