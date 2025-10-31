import numpy as np
import tensorflow as tf
import keras

from interface import MnistClassifierInterface

class NeuralNetwork(MnistClassifierInterface):
    def __init__(self, hidden_layer=64, activation='relu', learning_rate=0.001, batch_size=64, epochs=15):
        self.hidden_layer = hidden_layer # 64 seems to be balanced without overfitting mnist
        self.activation = activation # 'relu' prevents vanishing gradients, computationally efficietn
        self.learning_rate = learning_rate # 0.001 works good with Adam optimizer
        self.batch_size = batch_size # 64 balance between stability and efficiency
        self.epochs = epochs # 15 is sufficient for MNIST convergence without significant overfitting
        self.model = None   # will hold keras model
        self.history = None # will store training scores

        self._build_model()

    # underscore (_) in the function signature indicates its meant for private use
    def _build_model(self):
        # Architecture:
        # - Input layer: 784 neurons (flattened 28x28 MNIST images)
        # - Hidden layer: Dense layer with configurable neurons and activation (by default - 64 neurons)
        # - Dropout: Regularization layer to prevent overfitting
        # - Output layer: 10 neurons with softmax activation for classification
        self.model = keras.Sequential() # simplest model type

        self.model.add(keras.layers.Input(shape=(784,)))
        self.model.add(keras.layers.Dense(units=self.hidden_layer, activation=self.activation))
        self.model.add(keras.layers.Dropout(0.3)) # drop 30% of the neurons
        self.model.add(keras.layers.Dense(units=10, activation='softmax')) # to convert output to probabilites

        # specify how the model should learn
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'] # track accuracy during training
        )
        self.model.summary() # to see current model architecture

    def train(self, X_train, y_train, X_val, y_val):
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,     # number of samples per gradient update
            epochs=self.epochs,             # number of compelete passes through training dataset
            validation_data=(X_val, y_val), # validate loss after each epoch
            verbose=1, # turn on progress bar
            shuffle=True # mix up samples each epoch
        )

        return self.history
    
    def predict(self, X):
        probabilities = self.model.predict(X, verbose=0) # supress output

        # pick the highest probability out of all (0-9)
        predictions = np.argmax(probabilities, axis=1)

        return predictions
    
    def evaluate(self, X_test, y_test):
        # evaluates model on test data, check how it performs on unseen data
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        return test_accuracy