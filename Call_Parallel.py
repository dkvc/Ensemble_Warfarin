from scripts.bandits.ParallelThompson import ParallelThompsonSamplingBandit

#import cProfile
import pandas as pd
import pickle
import numpy as np

X_train, y_train = np.load('./data/cleaned_features.npy'), np.load('./data/cleaned_labels.npy')

def normalize(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-10)

X_train = normalize(X_train)

bins = pd.IntervalIndex.from_tuples([
    (0, 20.999),
    (20.999, 49),
    (49, 20000)
])

model = ParallelThompsonSamplingBandit(bins, num_models=10)

# cprofile across multiple pools
#cProfile.run('model.train(X_train, y_train, epochs=20)')
model.train(X_train, y_train, epochs=100)

model.save('./models/bandits/ParallelThompson.pkl')

# Try to load the model
with open('./models/bandits/ParallelThompson.pkl', 'rb') as file:
    model = pickle.load(file)

    from pprint import pprint
    pprint(f'ParallelThompson Accuracy: {model.score()}')
    pprint(f"Time taken: {model.time_taken:.3f}")
    pprint(f"ParallelThompson F1 Score: {model.f1_score()}")