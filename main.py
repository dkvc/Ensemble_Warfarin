import pickle
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts.baselines import *

base_path = "./models/baselines/"
baselines = ["FixedBaseLine", "ClinicalDosage", "PharmogenicDosage"]
models = baselines

accuracies = []
for model in models:
    with open(base_path + model + ".pkl", 'rb') as file:
            model = pickle.load(file)
            accuracy = model.score()
            accuracies += [accuracy]
            print(f"{model.__class__.__name__} Accuracy: {accuracy}")

# Plot models vs accuracies
plt.bar(models, accuracies, color='xkcd:sky blue')
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Model vs Accuracy")
plt.grid(True, alpha=0.3)
plt.yticks(np.arange(0, 1, 0.1))
plt.show()  