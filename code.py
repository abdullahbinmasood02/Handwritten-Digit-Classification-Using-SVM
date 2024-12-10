import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# SVM class with different kernels optimized for vectorization
class SVM:
    def __init__(self, kernel='linear', degree=3, gamma=0.01, learning_rate=0.001, lambda_param=0.01, n_iters=100):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _linear_kernel(self, X):
        return X @ self.w.T  # Vectorized linear kernel (dot product of X with weights)

    def _polynomial_kernel(self, X, X_i):
        return (1 + X @ X_i.T) ** self.degree  # Vectorized polynomial kernel

    def _rbf_kernel(self, X, X_i):
        dist = np.linalg.norm(X[:, np.newaxis] - X_i, axis=2)  # Vectorized RBF kernel
        return np.exp(-self.gamma * (dist ** 2))

    def _apply_kernel(self, X, X_i=None):
        if self.kernel == 'linear':
            return self._linear_kernel(X)
        elif self.kernel == 'polynomial':
            if X_i is None:
                X_i = X  # Use X as X_i if X_i is not provided
            return self._polynomial_kernel(X, X_i)
        elif self.kernel == 'rbf':
            if X_i is None:
                X_i = X  # Use X as X_i if X_i is not provided
            return self._rbf_kernel(X, X_i)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to +1 or -1

        for _ in range(self.n_iters):
            # Apply the kernel to all samples
            for idx in range(n_samples):
                # Pass the sample itself as X_i for the polynomial and RBF kernels
                condition = y_[idx] * (self._apply_kernel(X[idx].reshape(1, -1), X[idx].reshape(1, -1)) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(X[idx], y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        # Apply the kernel and make predictions based on weights and bias
        approx = self._apply_kernel(X) - self.b
        return np.sign(approx)  # Predict the class label based on the sign of the result


# Convert MNIST labels to binary (-1 for non-target, 1 for target)
def prepare_data_for_binary_classification(X, y, target_digit):
    y_binary = np.where(y == target_digit, 1, -1)
    return X, y_binary

# Extend SVM to multi-class using One-vs-Rest strategy
class SVM_OVR:
    def __init__(self, n_classes=10, kernel='linear', degree=3, gamma=0.01, learning_rate=0.001, lambda_param=0.01, n_iters=100):
        self.n_classes = n_classes
        self.models = [SVM(kernel, degree, gamma, learning_rate, lambda_param, n_iters) for _ in range(n_classes)]
    
    def fit(self, X, y):
        for i in range(self.n_classes):
            X_binary, y_binary = prepare_data_for_binary_classification(X, y, i)
            self.models[i].fit(X_binary, y_binary)
    
    def predict(self, X):
        # Vectorized prediction for each model
        predictions = np.array([model.predict(X) for model in self.models])  # Shape (n_classes, n_samples)
        return np.argmax(predictions, axis=0)  # Return class with the highest score for each sample


# Function to load MNIST image files
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.fromfile(f, dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
        return images

# Function to load MNIST label files
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

# Load train and test datasets
train_images = load_mnist_images('train-images.idx3-ubyte')
train_labels = load_mnist_labels('train-labels.idx1-ubyte')
test_images = load_mnist_images('t10k-images.idx3-ubyte')
test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')

# Use only the first 1000 samples for both training and testing
train_images_flat = train_images[:1000].reshape(1000, -1)  # First 1000 training images
train_labels_subset = train_labels[:1000]  # First 1000 training labels
test_images_flat = test_images[:1000].reshape(1000, -1)  # First 1000 test images
test_labels_subset = test_labels[:1000]  # First 1000 test labels
test_labels_subset = test_labels_subset.reshape(-1)
def evaluate_model(svm_ovr, kernel_name):
    predictions = svm_ovr.predict(test_images_flat)
    
    # Convert one-hot predictions (if any) to single label predictions
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Ensure that test_labels_subset has the same shape as predictions
    test_labels = test_labels_subset.flatten()

    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels)
    print(f"Kernel={kernel_name} -> Test Accuracy: {accuracy * 100:.2f}%")
    
    # Calculate confusion matrix, precision, recall, and F1-score
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, predictions))  # Ensure both are the same type/shape

    print("Classification Report:")
    print(classification_report(test_labels, predictions))

    # Identify misclassified images
    misclassified_indices = np.where(predictions != test_labels)[0]
    print(f"Number of misclassified images: {len(misclassified_indices)}")

    # Plot a few misclassified images
    num_images_to_show = min(10, len(misclassified_indices))  # Limit to 10 images
    plt.figure(figsize=(10, 10))

    for i, index in enumerate(misclassified_indices[:10]):
        plt.subplot(2, 5, i + 1)  # 2 rows, 5 columns for showing 10 images
        plt.imshow(test_images[index], cmap='gray')
        plt.title(f"True: {test_labels[index]}, Pred: {predictions[index]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Instantiate and train the multi-class SVM with linear kernel
print("Evaluating Linear Kernel")
svm_ovr_linear = SVM_OVR(n_classes=10, kernel='linear', learning_rate=0.001, lambda_param=0.01, n_iters=100)
svm_ovr_linear.fit(train_images_flat, train_labels_subset)
evaluate_model(svm_ovr_linear, 'Linear')

# Test different kernels and hyperparameters
print("Evaluating Polynomial Kernel (degree=2, 3, 4)")
degree_values = [2, 3, 4]
for degree in degree_values:
    svm_ovr_poly = SVM_OVR(n_classes=10, kernel='polynomial', degree=degree, learning_rate=0.001, lambda_param=0.01, n_iters=100)
    svm_ovr_poly.fit(train_images_flat, train_labels_subset)
    evaluate_model(svm_ovr_poly, f'Polynomial (degree={degree})')

print("Evaluating RBF Kernel (gamma=0.001, 0.01, 0.1)")
gamma_values = [0.001, 0.01, 0.1]
for gamma in gamma_values:
    svm_ovr_rbf = SVM_OVR(n_classes=10, kernel='rbf', gamma=gamma, learning_rate=0.001, lambda_param=0.01, n_iters=100)
    svm_ovr_rbf.fit(train_images_flat, train_labels_subset)
    evaluate_model(svm_ovr_rbf, f'RBF (gamma={gamma})')

# Test different values for regularization (lambda) with linear kernel
print("Evaluating Regularization (lambda)")
lambda_values = [0.001, 0.01, 10.0]
for lambda_param in lambda_values:
    print(f"Training model with lambda={lambda_param}...")
    svm_ovr_linear_reg = SVM_OVR(n_classes=10, kernel='linear', learning_rate=0.001, lambda_param=lambda_param, n_iters=100)
    svm_ovr_linear_reg.fit(train_images_flat, train_labels_subset)
    evaluate_model(svm_ovr_linear_reg, f'Linear (lambda={lambda_param})')
