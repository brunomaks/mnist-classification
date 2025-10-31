# MNIST Classification Project

A comprehensive implementation of handwritten digit classification using the MNIST dataset with three different machine learning approaches: Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network.

## 📋 Project Overview

This project demonstrates and compares the performance of three distinct machine learning models on the classic MNIST dataset, which contains 70,000 images of handwritten digits (0-9). The implementation provides a unified interface for training, evaluating, and comparing different classification approaches.

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

## 📊 Dataset

The MNIST dataset is automatically downloaded using `sklearn.datasets.fetch_openml`:
- **Training set**: 50,000 samples
- **Validation set**: 10,000 samples  
- **Test set**: 10,000 samples
- **Image size**: 28×28 pixels (784 features when flattened)
- **Classes**: 10 digits (0-9)

## 📈 Performance Metrics

Each model provides:
- **Training/Validation Accuracy**: Monitor for overfitting
- **Test Accuracy**: Final performance measure
- **Confidence Metrics** (Random Forest): Average and maximum confidence scores
- **Training History**: Loss curves for neural networks

## 📝 Key Findings

- **CNNs** typically achieve the highest accuracy due to their spatial feature extraction capabilities
- **Random Forests** provide interpretable confidence scores and fast training
- **Feed-Forward NNs** offer a good balance between performance and computational requirements
- Proper data preprocessing significantly impacts model performance

## 🎓 Skills Demonstrated

- **Machine Learning**: Random Forests, Neural Networks, CNNs using tensorflow
- **Data Preprocessing**: Normalization, reshaping, train/val/test splitting
- **Model Evaluation**: Accuracy metrics, overfitting detection
- **Software Engineering**: Modular design, clean code practices, documentation

---

**Note**: This project is designed for educational purposes and demonstrates practical implementation of machine learning concepts for computer vision tasks. The modular architecture makes it easy to extend and adapt for similar classification problems.
