import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from models.NeuralNetwork import NeuralNetwork
from models.RandomForest import RandomForest
from models.ConvolutionalNN import ConvolutionalNN

class MnistClassifier:
    def __init__(self, algorithm='nn'):
        self.algorithm = algorithm
        self.model = None

        if self.algorithm == 'nn':
            self.model = NeuralNetwork()
        elif self.algorithm == 'rf':  
            self.model = RandomForest()
        elif self.algorithm == 'cnn':
            self.model = ConvolutionalNN()
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

def preprocess_data_conv(X):
    # Add channel dimension (28,28) -> (28,28,1)  
    if len(X.shape) == 2 and X.shape[1] == 784:
        X = X.reshape(X.shape[0], 28, 28, 1)
    
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    
    return X

def plot_training_history(history, model_name):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = f'Training vs Validation Loss - {model_name}'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist_data()

    # X_train = preprocess_data(X_train)
    # X_val = preprocess_data(X_val)
    # X_test = preprocess_data(X_test)

    # # ===Testing Random Forest===
    # print("\nInitializing MnistClassifier with 'rf' algorithm")
    # classifier_rf = MnistClassifier(algorithm='rf')

    # print("\nTraining the model...")
    # history_rf = classifier_rf.train(X_train, y_train, X_val, y_val)

    # print("\nEvaluating on test set...")
    # test_metrics_rf = classifier_rf.evaluate(X_test, y_test)
    # print(f"\nFinal test accuracy: {test_metrics_rf['test_accuracy']:.4f} ({test_metrics_rf['test_accuracy']*100:.2f}%)")
    # print(f"Average confidence: {test_metrics_rf['test_average_confidence']:.4f}")
    # print(f"Max confidence: {test_metrics_rf['test_max_confidence']:.4f}")
    # print(f"Low confidence samples: {test_metrics_rf['test_low_confidence_count']}")

    # print(f"Training accuracy: {history_rf['train_accuracy']:.4f}")
    # print(f"Validation accuracy: {history_rf['val_accuracy']:.4f}")
    # print(f"Accuracy gap: {history_rf['accuracy_gap']:.4f}")

    # # ===Testing Neural Network===
    # print("\nInitializing MnistClassifier with 'nn' algorithm")
    # classifier_nn = MnistClassifier(algorithm='nn')

    # print("\nTraining the model...")
    # history_nn = classifier_nn.train(X_train, y_train, X_val, y_val)

    # print("\nEvaluating on test set...")
    # test_accuracy_nn = classifier_nn.evaluate(X_test, y_test)
    # print(f"\nFinal test accuracy: {test_accuracy_nn:.4f} ({test_accuracy_nn*100:.2f}%)")

    # # Plotting training & validation loss to check if the model is overfitting
    # plot_training_history(history=history_nn, model_name="NN")

    # ===Testing Convolutional Neural Network===
    X_train_conv = preprocess_data_conv(X_train)
    X_val_conv = preprocess_data_conv(X_val)
    X_test_conv = preprocess_data_conv(X_test)

    print(X_train_conv.shape)

    print("\nInitializing MnistClassifier with 'nn' algorithm")
    classifier_cnn = MnistClassifier(algorithm='cnn')

    print("\nTraining the model...")
    history_cnn = classifier_cnn.train(X_train_conv, y_train, X_val_conv, y_val)

    print("\nEvaluating on test set...")
    test_accuracy_cnn = classifier_cnn.evaluate(X_test_conv, y_test)
    print(f"\nFinal test accuracy: {test_accuracy_cnn:.4f} ({test_accuracy_cnn*100:.2f}%)")

    # Plotting training & validation loss to check if the model is overfitting
    plot_training_history(history=history_cnn, model_name="CNN")


    # Comparing predictions between all the models
    # print("\nMaking predictions on first 5 test samples...")
    # predictions_rf = classifier_rf.predict(X_test[:5])
    # print(f"RF Predictions: {predictions_rf}")

    # predictions_nn = classifier_nn.predict(X_test[:5])
    # print(f"NN Predictions: {predictions_nn}")

    predictions_cnn = classifier_cnn.predict(X_test_conv[:5])
    print(f"CNN Predictions: {predictions_cnn}")

    print(f"Actual labels: {y_test[:5]}")

if __name__ == "__main__":
    main()