import os

import numpy as np
import pandas as pd

bins = pd.IntervalIndex.from_tuples([
    (0, 20.999),
    (20.999, 49),
    (49, 20000)
])

def load_data(df):
    df = df.dropna()
    df['Dose_Bin'] = pd.cut(df['Therapeutic Dose of Warfarin'].astype(float), bins=bins, labels=False)

    labels = df['Dose_Bin']
    features = df.drop(columns=['Therapeutic Dose of Warfarin', 'Dose_Bin'])
    return features, labels

def extract_all_features(df):
    features = []
    # Age in decades
    features.append(df['Age'].astype(float))

    # Height in cm
    features.append(df['Height (cm)'].astype(float))

    # Weight in kg
    features.append(df['Weight (kg)'].astype(float))

    # Amiodarone status (converted to float for binary representation)
    features.append(df['Amiodarone (Cordarone)'].astype(float))

    # Enzyme inducer status
    features.append(df['Enzyme_inducer'].astype(float))

    race_columns = ['Race_Asian', 'Race_Black or African American', 'Race_White', 'Race_Unknown']
    race_df = df[race_columns]
    features.append(race_df.values)

    # CYP2C9 genotypes (assuming columns range from 'CYP2C9_*' names)
    cyp2c9_columns = [col for col in df.columns if 'CYP2C9' in col]
    features.append(df[cyp2c9_columns].astype(int))

    # VKORC1 genotypes (assuming columns range from 'VKORC1_*' names)
    vkorc1_columns = [col for col in df.columns if 'VKORC1' in col]
    features.append(df[vkorc1_columns].astype(int))

    # Add bias (constant 1)
    features.append(np.ones(len(df)))

    # Convert list of features into a single NumPy array
    return np.hstack([np.array(f).reshape(-1, 1) if f.ndim == 1 else f for f in features])

def clean(dataset):
    features_df, labels = load_data(dataset)
    features = extract_all_features(features_df)

    labels = bins.get_indexer(labels)
    return features, labels

def save(features, labels):
    dataset_dir_location = './data/modified'
    os.makedirs(dataset_dir_location, exist_ok=True)

    features_location = dataset_dir_location + "/deep_cleaned_features.npy"
    labels_location = dataset_dir_location + "/deep_cleaned_labels.npy"

    print("Saving features array to", features_location)
    np.save(features_location, features)

    print("Saving labels array to", labels_location)
    np.save(labels_location, labels)

if __name__ == "__main__":
    dataset = pd.read_csv('./data/modified/simple_cleaned.csv')
    features, labels = clean(dataset)
    save(features, labels)