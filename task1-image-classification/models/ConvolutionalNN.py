import numpy as np
import tensorflow as tf
import keras

from interface import MnistClassifierInterface

class ConvolutionalNN(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, learning_rate=0.001, batch_size=32, epochs=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.history = None
    
        self._build_model()

    def _build_model(self):
        self.model = keras.Sequential([
            # convolutional layer - learn spatial features
            keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=self.input_shape),
            # pooling layer - reduce the size, keep important features
            keras.layers.MaxPooling2D((2,2)),
            # flatten - convert 2d features to 1d for classification
            keras.layers.Flatten(),
            # final classification layer - use softmax to predict probability
            keras.layers.Dense(10, activation='softmax')
       ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'] # track accuracy during training
        )

        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val):
        self.history = self.model.fit(X_train, 
                                 y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=(X_val, y_val),
                                 verbose=1)
        
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
    
