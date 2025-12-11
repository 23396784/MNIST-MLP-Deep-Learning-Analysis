"""
Data Loader Module for MNIST Neural Network Analysis
=====================================================
Handles MNIST dataset loading, preprocessing, and preparation for training.

Author: Victor Prefa
Course: SIG720 Machine Learning, Deakin University
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_mnist(normalize: bool = True, one_hot: bool = True) -> tuple:
    """
    Load and preprocess the MNIST dataset.
    
    Parameters
    ----------
    normalize : bool
        Whether to normalize pixel values to [0, 1]
    one_hot : bool
        Whether to convert labels to one-hot encoding
        
    Returns
    -------
    tuple
        ((X_train, y_train), (X_test, y_test))
    """
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(f"Original shapes:")
    print(f"  Training: {X_train.shape}, Labels: {y_train.shape}")
    print(f"  Test: {X_test.shape}, Labels: {y_test.shape}")
    
    # Normalize pixel values
    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        print("  Normalized pixel values to [0, 1]")
    
    # One-hot encode labels
    if one_hot:
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        print("  Converted labels to one-hot encoding")
    
    print(f"\nFinal shapes:")
    print(f"  Training: {X_train.shape}, Labels: {y_train.shape}")
    print(f"  Test: {X_test.shape}, Labels: {y_test.shape}")
    
    return (X_train, y_train), (X_test, y_test)


def flatten_images(X: np.ndarray) -> np.ndarray:
    """
    Flatten 28x28 images to 784-dimensional vectors.
    
    Parameters
    ----------
    X : np.ndarray
        Image array of shape (n_samples, 28, 28)
        
    Returns
    -------
    np.ndarray
        Flattened array of shape (n_samples, 784)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def get_data_info(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Get comprehensive information about the dataset.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
        
    Returns
    -------
    dict
        Dataset information
    """
    info = {
        'n_train_samples': X_train.shape[0],
        'n_test_samples': X_test.shape[0],
        'image_shape': X_train.shape[1:],
        'n_features': np.prod(X_train.shape[1:]),
        'n_classes': y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(y_train)),
        'train_test_ratio': X_train.shape[0] / X_test.shape[0],
        'pixel_range': (X_train.min(), X_train.max())
    }
    
    print("=" * 50)
    print("MNIST DATASET INFORMATION")
    print("=" * 50)
    print(f"Training samples: {info['n_train_samples']:,}")
    print(f"Test samples: {info['n_test_samples']:,}")
    print(f"Image shape: {info['image_shape']}")
    print(f"Features (flattened): {info['n_features']}")
    print(f"Number of classes: {info['n_classes']}")
    print(f"Train/Test ratio: {info['train_test_ratio']:.1f}:1")
    print(f"Pixel value range: {info['pixel_range']}")
    print("=" * 50)
    
    return info


def visualize_samples(X: np.ndarray, y: np.ndarray, n_samples: int = 10):
    """
    Visualize sample images from the dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Image array
    y : np.ndarray
        Label array
    n_samples : int
        Number of samples to display
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 2))
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Handle flattened or 2D images
        if len(X.shape) == 2:
            img = X[i].reshape(28, 28)
        else:
            img = X[i]
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        # Handle one-hot or integer labels
        if len(y.shape) > 1:
            label = np.argmax(y[i])
        else:
            label = y[i]
        
        ax.set_title(f'{label}')
    
    plt.suptitle('MNIST Sample Images', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    (X_train, y_train), (X_test, y_test) = load_mnist()
    info = get_data_info(X_train, y_train, X_test, y_test)
    visualize_samples(X_train, y_train)
