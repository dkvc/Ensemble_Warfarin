import pickle
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts.baselines import *
from scripts.bandits import *

base_path = "./models/"
baselines = ["baselines/FixedBaseLine", "baselines/ClinicalDosage", "baselines/PharmogenicDosage"]
bandits = ["bandits/LinUCB", "bandits/Lasso", "bandits/Thompson", "bandits/Ensemble"]
models = baselines + bandits

def normalize(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-10)

accuracies = []
f1_scores = []
for model_name in models:
    with open(base_path + model_name + ".pkl", 'rb') as file:
            model = pickle.load(file)
            accuracy = model.score()

            accuracies += [accuracy]
            if hasattr(model, 'time_taken'):
                print(f"{model.__class__.__name__}, Accuracy: {accuracy}, Time taken: {model.time_taken:.3f} seconds")
            else:
                print(f"{model.__class__.__name__}, Accuracy: {accuracy}")

            if model_name in bandits:
                f1_score = model.f1_score()
                f1_scores += [f1_score]

                print(f"{model.__class__.__name__}, F1 Score: {f1_score}")

# Plot models vs accuracies
plt.bar(models, accuracies, color='xkcd:sky blue')
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Model vs Accuracy")
plt.grid(True, alpha=0.3)
plt.yticks(np.arange(0, 1, 0.1))
plt.show()

# Plot models vs f1 scores
plt.bar(bandits, f1_scores, color='xkcd:lavender')
plt.ylabel("F1 Score")
plt.xlabel("Model")
plt.title("Model vs F1 Score")
plt.grid(True, alpha=0.3)
plt.yticks(np.arange(0, 1, 0.1))
plt.show()