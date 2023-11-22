#-------------------------------------------------------------------------
# AUTHOR: Vu Nguyen
# FILENAME: Deep Learning
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU CAN USE ANY PYTHON LIBRARY TO COMPLETE YOUR CODE.

#importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def build_model(n_hidden, n_neurons_hidden, n_neurons_output, learning_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))  # input layer

    # iterate over the number of hidden layers to create the hidden layers:
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons_hidden, activation="relu"))

    # output layer
    model.add(keras.layers.Dense(n_neurons_output, activation="softmax"))

    # defining the learning rate
    opt = keras.optimizers.SGD(learning_rate)

    # Compiling the Model specifying the loss function and the optimizer to use.
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

# Load the dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Creating a validation set and scaling the features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Hyperparameters
n_hidden = [2, 5, 10]
n_neurons = [10, 50, 100]
l_rate = [0.01, 0.05, 0.1]

highestAccuracy = 0
best_model = None
best_params = {}

# Iterating over hyperparameters
for h in n_hidden:
    for n in n_neurons:
        for l in l_rate:
            model = build_model(h, n, 10, l)  # 10 output neurons for 10 classes

            # Training the model
            history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))

            # Evaluate the model
            test_loss, test_acc = model.evaluate(X_test, y_test)

            if test_acc > highestAccuracy:
                highestAccuracy = test_acc
                best_model = model
                best_params = {'hidden_layers': h, 'neurons': n, 'learning_rate': l}

            print(f"Highest accuracy so far: {highestAccuracy}")
            print(f"Parameters: Number of Hidden Layers: {h}, number of neurons: {n}, learning rate: {l}")
            print()

# Best model summary
print(best_model.summary())

# Plotting the learning curves
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# # Save model architecture plot
# img_file = './model_arch.png'
# tf.keras.utils.plot_model(best_model, to_file=img_file, show_shapes=True, show_layer_names=True)




