"""
MLP Models Module for MNIST Neural Network Analysis
====================================================
Implements various Multi-Layer Perceptron architectures for experiments.

Author: Victor Prefa
Course: SIG720 Machine Learning, Deakin University
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


def create_single_layer_mlp(hidden_size: int = 128, 
                            input_shape: tuple = (28, 28),
                            num_classes: int = 10) -> Sequential:
    """
    Create a single hidden layer MLP model.
    
    Parameters
    ----------
    hidden_size : int
        Number of neurons in hidden layer
    input_shape : tuple
        Input image shape
    num_classes : int
        Number of output classes
        
    Returns
    -------
    Sequential
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(hidden_size, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Created Single Hidden Layer MLP:")
    print(f"  Architecture: Input({np.prod(input_shape)}) → Dense({hidden_size}) → Output({num_classes})")
    print(f"  Total Parameters: {model.count_params():,}")
    
    return model


def create_mlp(hidden_layers: list, 
               input_shape: tuple = (28, 28),
               num_classes: int = 10,
               activation: str = 'relu') -> Sequential:
    """
    Create a multi-layer perceptron with configurable depth.
    
    Parameters
    ----------
    hidden_layers : list
        List of hidden layer sizes, e.g., [100, 100] for 2 layers
    input_shape : tuple
        Input image shape
    num_classes : int
        Number of output classes
    activation : str
        Activation function for hidden layers
        
    Returns
    -------
    Sequential
        Compiled Keras model
    """
    layers = [
        Input(shape=input_shape),
        Flatten()
    ]
    
    for i, units in enumerate(hidden_layers):
        layers.append(Dense(units, activation=activation, name=f'hidden_{i+1}'))
    
    layers.append(Dense(num_classes, activation='softmax', name='output'))
    
    model = Sequential(layers)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print architecture
    arch_str = f"Input({np.prod(input_shape)})"
    for i, units in enumerate(hidden_layers):
        arch_str += f" → Dense({units})"
    arch_str += f" → Output({num_classes})"
    
    print(f"Created {len(hidden_layers)}-Layer MLP:")
    print(f"  Architecture: {arch_str}")
    print(f"  Total Parameters: {model.count_params():,}")
    
    return model


def create_depth_experiment_model(num_layers: int,
                                  layer_size: int = 100,
                                  input_shape: tuple = (28, 28),
                                  num_classes: int = 10) -> Sequential:
    """
    Create MLP for depth analysis experiment.
    
    Parameters
    ----------
    num_layers : int
        Number of hidden layers
    layer_size : int
        Size of each hidden layer (constant)
    input_shape : tuple
        Input image shape
    num_classes : int
        Number of output classes
        
    Returns
    -------
    Sequential
        Compiled Keras model
    """
    hidden_layers = [layer_size] * num_layers
    return create_mlp(hidden_layers, input_shape, num_classes)


def create_width_experiment_model(layer_size: int,
                                  input_shape: tuple = (28, 28),
                                  num_classes: int = 10) -> Sequential:
    """
    Create single-layer MLP for width analysis experiment.
    
    Parameters
    ----------
    layer_size : int
        Size of the hidden layer
    input_shape : tuple
        Input image shape
    num_classes : int
        Number of output classes
        
    Returns
    -------
    Sequential
        Compiled Keras model
    """
    return create_single_layer_mlp(layer_size, input_shape, num_classes)


def create_double_descent_model(hidden_size_1: int,
                                hidden_size_2: int,
                                input_shape: tuple = (28, 28),
                                num_classes: int = 10) -> Sequential:
    """
    Create two-layer MLP for double descent investigation.
    
    Parameters
    ----------
    hidden_size_1 : int
        Size of first hidden layer
    hidden_size_2 : int
        Size of second hidden layer
    input_shape : tuple
        Input image shape
    num_classes : int
        Number of output classes
        
    Returns
    -------
    Sequential
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(hidden_size_1, activation='relu', name='hidden_1'),
        Dense(hidden_size_2, activation='relu', name='hidden_2'),
        Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Created Double Descent Model: ({hidden_size_1}, {hidden_size_2})")
    print(f"  Total Parameters: {model.count_params():,}")
    
    return model


def get_early_stopping(patience: int = 3, 
                       monitor: str = 'val_loss',
                       restore_best: bool = True) -> EarlyStopping:
    """
    Create early stopping callback.
    
    Parameters
    ----------
    patience : int
        Number of epochs with no improvement to wait
    monitor : str
        Metric to monitor
    restore_best : bool
        Whether to restore best weights
        
    Returns
    -------
    EarlyStopping
        Configured callback
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=restore_best,
        verbose=1
    )


def count_parameters(model: Sequential) -> dict:
    """
    Get detailed parameter count by layer.
    
    Parameters
    ----------
    model : Sequential
        Keras model
        
    Returns
    -------
    dict
        Parameter counts by layer
    """
    layer_params = {}
    total = 0
    
    for layer in model.layers:
        params = layer.count_params()
        layer_params[layer.name] = params
        total += params
    
    layer_params['total'] = total
    
    return layer_params


if __name__ == "__main__":
    # Test model creation
    print("=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    # Single layer
    model1 = create_single_layer_mlp(128)
    print()
    
    # Multi-layer
    model2 = create_mlp([100, 100])
    print()
    
    # Depth experiment
    model3 = create_depth_experiment_model(4, layer_size=100)
    print()
    
    # Double descent
    model4 = create_double_descent_model(500, 500)
