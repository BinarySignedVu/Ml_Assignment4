#-------------------------------------------------------------------------
# AUTHOR: Vu Nguyen
# FILENAME: Deep Learning
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

# Hyperparameters
n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

# Load datasets
df = pd.read_csv('../../Users/hyper/OneDrive/Desktop/machine learning/4120 - ML/HW 4/Prompt/optdigits.tra', sep=',', header=None)
X_training = np.array(df.values)[:,:64]
y_training = np.array(df.values)[:,-1]

df = pd.read_csv('../../Users/hyper/OneDrive/Desktop/machine learning/4120 - ML/HW 4/Prompt/optdigits.tes', sep=',', header=None)
X_test = np.array(df.values)[:,:64]
y_test = np.array(df.values)[:,-1]

# Track the best performance
highest_accuracy_perceptron = 0
highest_accuracy_mlp = 0
best_params_perceptron = {}
best_params_mlp = {}

# Iterate over hyperparameters
for lr in n:
    for shuffle in r:
        # Perceptron
        perceptron = Perceptron(eta0=lr, shuffle=shuffle, max_iter=1000)
        perceptron.fit(X_training, y_training)
        perceptron_accuracy = perceptron.score(X_test, y_test)
        if perceptron_accuracy > highest_accuracy_perceptron:
            highest_accuracy_perceptron = perceptron_accuracy
            best_params_perceptron = {'learning rate': lr, 'shuffle': shuffle}

        # MLPClassifier
        mlp = MLPClassifier(activation='logistic', learning_rate_init=lr, hidden_layer_sizes=(100,), shuffle=shuffle, max_iter=1000)
        mlp.fit(X_training, y_training)
        mlp_accuracy = mlp.score(X_test, y_test)
        if mlp_accuracy > highest_accuracy_mlp:
            highest_accuracy_mlp = mlp_accuracy
            best_params_mlp = {'learning rate': lr, 'shuffle': shuffle}

# Print best accuracies and parameters
print(f"Highest Perceptron accuracy so far: {highest_accuracy_perceptron}, Parameters: {best_params_perceptron}")
print(f"Highest MLP accuracy so far: {highest_accuracy_mlp}, Parameters: {best_params_mlp}")












