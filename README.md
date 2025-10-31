# MNIST Classification Project

A comprehensive implementation of handwritten digit classification using the MNIST dataset with three different machine learning approaches: Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network.

## 📋 Project Overview

This project demonstrates and compares the performance of three distinct machine learning models on the classic MNIST dataset, which contains 70,000 images of handwritten digits (0-9). The implementation provides a unified interface for training, evaluating, and comparing different classification approaches.

## 🚀 Features

- **Multiple Model Architectures**:
  - Random Forest (Traditional ML)
  - Feed-Forward Neural Network (Deep Learning)
  - Convolutional Neural Network (Computer Vision)

- **Modular Design**: Clean separation of model implementations
- **Comprehensive Evaluation**: Training history, accuracy metrics, and confidence analysis
- **Data Preprocessing**: Specialized preprocessing pipelines for different model types
- **Visualization**: Training progress and performance metrics plotting

## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.7+
- pip package manager

### Required Libraries
```bash
pip install numpy matplotlib scikit-learn tensorflow
```

## 📁 Project Structure

```
mnist-classifier/
├── models/
│   ├── __init__.py
│   ├── NeuralNetwork.py      # Feed-Forward NN implementation
│   ├── RandomForest.py       # Random Forest implementation
│   └── ConvolutionalNN.py    # CNN implementation
├── main.py                   # Main training and evaluation script
└── README.md
```

## 🧠 Model Architectures

### 1. Random Forest (`RandomForest.py`)
- Ensemble method using multiple decision trees
- Handles 784-dimensional flattened image data
- Provides confidence scores for predictions
- No feature scaling required

### 2. Feed-Forward Neural Network (`NeuralNetwork.py`)
- Multi-layer perceptron architecture
- Input layer: 784 neurons (28×28 pixels)
- Hidden layers with ReLU activation
- Output layer: 10 neurons with softmax activation
- Uses categorical cross-entropy loss

### 3. Convolutional Neural Network (`ConvolutionalNN.py`)
- Specialized for image data
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- Preserves spatial relationships in images

## 📊 Dataset

The MNIST dataset is automatically downloaded using `sklearn.datasets.fetch_openml`:
- **Training set**: 50,000 samples
- **Validation set**: 10,000 samples  
- **Test set**: 10,000 samples
- **Image size**: 28×28 pixels (784 features when flattened)
- **Classes**: 10 digits (0-9)

## 🎯 Usage

### Basic Usage
```python
from main import MnistClassifier, load_mnist_data

# Load and preprocess data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist_data()

# Initialize classifier (choose 'rf', 'nn', or 'cnn')
classifier = MnistClassifier(algorithm='cnn')

# Train the model
history = classifier.train(X_train, y_train, X_val, y_val)

# Evaluate on test set
accuracy = classifier.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

### Running the Complete Pipeline
```bash
python main.py
```

The main script includes:
- Data loading and preprocessing
- Model training for all architectures
- Performance evaluation
- Training history visualization
- Sample predictions

## 📈 Performance Metrics

Each model provides:
- **Training/Validation Accuracy**: Monitor for overfitting
- **Test Accuracy**: Final performance measure
- **Confidence Metrics** (Random Forest): Average and maximum confidence scores
- **Training History**: Loss curves for neural networks

## 🔧 Customization

### Adding New Models
Extend the framework by implementing new model classes with the interface:
```python
class CustomModel:
    def train(self, X_train, y_train, X_val, y_val):
        # Training logic
        return history
    
    def predict(self, X):
        # Prediction logic
        return predictions
    
    def evaluate(self, X_test, y_test):
        # Evaluation logic
        return accuracy
```

### Hyperparameter Tuning
Modify hyperparameters in respective model files:
- Random Forest: n_estimators, max_depth
- Neural Network: layers, units, learning rate
- CNN: filters, kernel size, pooling strategy

## 📝 Key Findings

- **CNNs** typically achieve the highest accuracy due to their spatial feature extraction capabilities
- **Random Forests** provide interpretable confidence scores and fast training
- **Feed-Forward NNs** offer a good balance between performance and computational requirements
- Proper data preprocessing significantly impacts model performance

## 🎓 Skills Demonstrated

- **Machine Learning**: Random Forests, Neural Networks, CNNs
- **Deep Learning**: TensorFlow/Keras, model architecture design
- **Data Preprocessing**: Normalization, reshaping, train/val/test splitting
- **Model Evaluation**: Accuracy metrics, overfitting detection, confidence analysis
- **Software Engineering**: Modular design, clean code practices, documentation

## 🔮 Future Enhancements

- Hyperparameter optimization using GridSearchCV or Bayesian optimization
- Data augmentation for improved generalization
- Ensemble methods combining multiple models
- Real-time digit classification web interface
- Transfer learning with pre-trained models

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Note**: This project is designed for educational purposes and demonstrates practical implementation of machine learning concepts for computer vision tasks. The modular architecture makes it easy to extend and adapt for similar classification problems.
