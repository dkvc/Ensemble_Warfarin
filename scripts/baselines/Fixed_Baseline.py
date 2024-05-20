# TODO: Refactor this code as a class
import pickle
import pandas as pd

class FixedBaseLine:
    def __init__(self, dataset, bins, dropna=True):
        if dropna:
            dataset.dropna(inplace=True)
        self.bins = bins
        self.cut = pd.cut(dataset['Therapeutic Dose of Warfarin'], bins)

    def predict(self):
        return self.bins[1]

    def score(self):
        return sum(self.cut == self.bins[1]) / len(self.cut)
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

if __name__ == "__main__":
    dataset = pd.read_csv('./data/cleaned.csv')

    """
    splits:
    < 21mg/week
    21-49mg/week
    > 49mg/week
    """

    # split
    bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49, 20000)
    ])

    cut = pd.cut(dataset['Therapeutic Dose of Warfarin'], bins)
    nums = dataset['Therapeutic Dose of Warfarin'].value_counts(bins=bins).sort_index()
    # print(nums)

    # # fixed baseline: medium class (21-49mg/week)
    # baseline = nums[bins[1]]
    # print(f"Accuracy: {baseline / sum(nums)}")
    model = FixedBaseLine(dataset, bins)
    print(f"Accuracy: {model.score()}")

    model.save('./models/baselines/FixedBaseLine.pkl')

    # try to load
    with open('./models/baselines/FixedBaseLine.pkl', 'rb') as file:
        model = pickle.load(file)
        print(f"Accuracy: {model.score()}")