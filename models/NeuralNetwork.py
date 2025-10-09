import numpy as np
import tensorflow as tf
import keras

from interface import MnistClassifierInterface

class NeuralNetwork(MnistClassifierInterface):
    def __init__(self, hidden_layers=[128, 64], activation='relu', learning_rate=0.001, batch_size=32, epochs=15):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None   # will hold keras model
        self.history = None # will store training scores

        self._build_model()

    def _build_model(self):
        self.model = keras.Sequential() # simplest model type

        self.model.add(keras.layers.Input(shape=(784,))) # set input layer to take 784 inputs (because of 28x28 mnist image)
        self.model.add(keras.layers.Dense(units=self.hidden_layers[0], activation=self.activation))
        self.model.add(keras.layers.Dense(units=self.hidden_layers[1], activation=self.activation))
        self.model.add(keras.layers.Dense(units=10, activation='softmax')) # to convert output to probabilites

        # specify how the model should learn
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.summary() # to see current model architecture

    def preprocess_data(self, X):
        # normalization
        X = X.astype('float32') / 255.0

        # converting 2d images to 1d arrays
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        return X
    
    def train(self, X_train, y_train, X_val, y_val):
        X_train = self.preprocess_data(X_train)
        X_val = self.preprocess_data(X_val)

        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            verbose=1,
            shuffle=True # mix up samples each epoch
        )

        return self.history
    
    def predict(self, X):
        X = self.preprocess_data(X)

        probabilities = self.model.predict(X, verbose=0)

        predictions = np.argmax(probabilities, axis=1)

        return predictions
    
    def evaluate(self, X_test, y_test):
        X_test = self.preprocess_data(X_test)

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        return test_accuracy