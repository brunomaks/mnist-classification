import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from interface import MnistClassifierInterface

class RandomForest(MnistClassifierInterface):
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        self.n_estimators = n_estimators # number of trees in the forest
        self.max_depth = max_depth # max depth of the trees
        self.random_state = random_state # seed for reproducibility
        self.model = None # to hold the model

    def train(self, X_train, y_train, X_val, y_val):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, 
                                            max_depth=self.max_depth, 
                                            random_state=self.random_state)

        self.model.fit(X_train, y_train)

        train_predictions = self.predict(X_train)
        val_predictions = self.predict(X_val)

        train_accuracy = accuracy_score(y_train, train_predictions)
        train_loss = 1 - train_accuracy

        val_accuracy = accuracy_score(y_val, val_predictions)
        val_loss = 1 - val_accuracy

        accuracy_gap = train_accuracy - val_accuracy

        return { 'train_accuracy': train_accuracy,
                 'train_loss': train_loss,
                 'val_accuracy': val_accuracy,
                 'val_loss': val_loss,
                 'accuracy_gap': accuracy_gap }

    def predict(self, X):
        return self.model.predict(X)
    
    def _predict_probabilities(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        test_predictions = self.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_loss = 1 - test_accuracy

        test_probabilities = self._predict_probabilities(X_test)
        max_confidences = np.max(test_probabilities, axis=1)

        return { 
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'test_average_confidence': np.mean(max_confidences),
            'test_max_confidence': np.max(max_confidences),
            'test_min_confidence': np.min(max_confidences),
            'test_confidence_std': np.std(max_confidences),
            'test_low_confidence_count': np.sum(max_confidences < 0.6)
        }
    