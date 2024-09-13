import pickle
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts.baselines import *
from scripts.bandits import *

base_path = "./models/"
baselines = ["baselines/FixedBaseLine", "baselines/ClinicalDosage", "baselines/PharmogenicDosage"]
bandits = ["bandits/LinUCB", "bandits/Lasso"]
torch_bandits = ["bandits/Thompson", "bandits/EnsembleSampling"]
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

for model_name in torch_bandits:
    X_train, y_train = np.load('./data/cleaned_features.npy'), np.load('./data/cleaned_labels.npy')
    X_train = normalize(X_train)
    if model_name == "bandits/EnsembleSampling":
        model = torch.load(base_path + model_name + ".pkl")
        acc = 0
        f1 = 0
        for i in range(len(model)):
            model[i].eval()  # Set the model to eval mode
            with torch.no_grad():  # Disable gradients
                y_pred = [torch.argmax(model[i](torch.tensor(X_train[j], dtype=torch.float32).to('cuda'))).item() for j in range(len(X_train))]
                acc += model[i].score(y_train, y_pred)
                f1 += model[i].f1_score(y_train, y_pred)
        accuracies += [acc / len(model)]
        f1_scores += [f1 / len(model)]
        print(f"{model.__class__.__name__}, Accuracy: {acc / len(model)}")
    else:
        import pandas as pd
        bins = pd.IntervalIndex.from_tuples([
        (0, 20.999),
        (20.999, 49),
        (49 ,20000)
        ])
        model = ThompsonSamplingBandit(bins, device='cuda')
        model.load(base_path + model_name + ".pkl")
        accuracy = model.score()
        accuracies += [accuracy]
        print(f"{model.__class__.__name__}, Accuracy: {accuracy}")

    if hasattr(model, 'time_taken'):
        print(f"{model.__class__.__name__}, Time taken: {model.time_taken:.3f} seconds")
    
    if model_name == "bandits/EnsembleSampling":
        print(f"{model.__class__.__name__}, Time taken: {model.time_taken():.3f} seconds")
        print(f"EnsembleSampling, F1 Score: {f1 / len(model)}")
    else:
        f1_score = model.f1_score()
        f1_scores += [f1_score]
        print(f"ThompsonSamplingBandit, F1 Score: {f1_score}")

models += torch_bandits

# Plot models vs accuracies
plt.bar(models, accuracies, color='xkcd:sky blue')
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Model vs Accuracy")
plt.grid(True, alpha=0.3)
plt.yticks(np.arange(0, 1, 0.1))
plt.show()

bandits += torch_bandits

# Plot models vs f1 scores
plt.bar(bandits, f1_scores, color='xkcd:lavender')
plt.ylabel("F1 Score")
plt.xlabel("Model")
plt.title("Model vs F1 Score")
plt.grid(True, alpha=0.3)
plt.yticks(np.arange(0, 1, 0.1))
plt.show()