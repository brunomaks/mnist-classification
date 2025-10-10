import matplotlib.pyplot as plt
from models.NeuralNetwork import NeuralNetwork
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class MnistClassifier:
    def __init__(self, algorithm='nn'):
        self.algorithm = algorithm
        self.model = None

        if self.algorithm == 'nn':
            self.model = NeuralNetwork()
        else:
            raise ValueError(f"Uknown algorithm: {algorithm}. Choose from 'nn', 'cnn' or 'rf'")
        
        
    def train(self, X_train, y_train, X, y):
        return self.model.train(X_train, y_train, X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    # X_train: (60 000, 784)
    # X_test:  (10 000, 784)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=10000, random_state=42, stratify=y_train
    )
    # X_train: (50 000, 784)
    # X_val:   (10 000, 784)
    # X_test:  (10 000, 784)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_data(X):
    # normalization
    X = X.astype('float32') / 255.0

    # converting 2d images to 1d arrays
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    return X

def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist_data()

    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)
    X_test = preprocess_data(X_test)

    print("\nInitializing MnistClassifier with 'nn' algorithm")
    classifier = MnistClassifier(algorithm='nn')

    print("\nTraining the model (preprocessing included)...")
    history = classifier.train(X_train, y_train, X_val, y_val)

    print("\nEvaluating on test set...")
    test_accuracy = classifier.evaluate(X_test, y_test)
    print(f"\nFinal test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    print("\nMaking predictions on first 5 test samples...")
    predictions = classifier.predict(X_test[:5])
    print(f"Predictions: {predictions}")
    print(f"Actual labels: {y_test[:5]}")

    # Plotting Training & Validation loss to check if the model is overfitting
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs validation loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()