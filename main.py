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

accuracies = []
for model in models:
    with open(base_path + model + ".pkl", 'rb') as file:
            model = pickle.load(file)
            accuracy = model.score()
            accuracies += [accuracy]
            if hasattr(model, 'time_taken'):
                print(f"{model.__class__.__name__}, Accuracy: {accuracy}, Time taken: {model.time_taken:.3f} seconds")
            else:
                print(f"{model.__class__.__name__}, Accuracy: {accuracy}")

# Plot models vs accuracies
plt.bar(models, accuracies, color='xkcd:sky blue')
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Model vs Accuracy")
plt.grid(True, alpha=0.3)
plt.yticks(np.arange(0, 1, 0.1))
plt.show()  