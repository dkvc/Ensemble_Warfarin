from scripts.bandits.Ensemble import EnsembleSamplingBandit

#import cProfile
import pandas as pd
import pickle
train_df, test_df = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
X_train, y_train = train_df.drop(columns=['Therapeutic Dose of Warfarin']), train_df['Therapeutic Dose of Warfarin']

bins = pd.IntervalIndex.from_tuples([
    (0, 20.999),
    (20.999, 49),
    (49, 20000)
])

model = EnsembleSamplingBandit(train_df, bins)

# cprofile across multiple pools
#cProfile.run('model.train(X_train, y_train, epochs=20)')
model.train(X_train, y_train, epochs=20)

model.save('./models/bandits/Ensemble.pkl')

# Try to load the model
with open('./models/bandits/Ensemble.pkl', 'rb') as file:
    model = pickle.load(file)

    from pprint import pprint
    pprint(f'Ensemble Accuracy: {model.score()}')
    pprint(f"Time taken: {model.time_taken:.3f}")