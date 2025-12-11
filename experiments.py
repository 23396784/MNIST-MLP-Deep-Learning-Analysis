"""
Experiments Module for MNIST Neural Network Analysis
=====================================================
Implements depth, width, and double descent experiments.

Author: Victor Prefa
Course: SIG720 Machine Learning, Deakin University
"""

import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from .mlp_models import (
    create_depth_experiment_model,
    create_width_experiment_model,
    create_double_descent_model,
    get_early_stopping
)


def run_depth_experiment(X_train, y_train, X_test, y_test,
                         layer_counts: list = [2, 4, 6, 8, 10],
                         layer_size: int = 100,
                         epochs: int = 15,
                         batch_size: int = 128) -> pd.DataFrame:
    """
    Investigate effect of network depth on performance.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    layer_counts : list
        Number of hidden layers to test
    layer_size : int
        Size of each hidden layer
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
        
    Returns
    -------
    pd.DataFrame
        Experiment results
    """
    print("=" * 60)
    print("DEPTH ANALYSIS EXPERIMENT")
    print(f"Testing {len(layer_counts)} configurations with {layer_size} neurons per layer")
    print("=" * 60)
    
    results = []
    early_stopping = get_early_stopping(patience=3)
    
    for num_layers in layer_counts:
        print(f"\n🧪 Testing {num_layers}-layer MLP...")
        
        # Create model
        model = create_depth_experiment_model(num_layers, layer_size)
        total_params = model.count_params()
        
        # Train
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0
        )
        training_time = time.time() - start_time
        converged_epoch = len(history.history['accuracy'])
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        results.append({
            'num_layers': num_layers,
            'total_params': total_params,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'generalization_gap': train_acc - test_acc,
            'training_time': training_time,
            'converged_epoch': converged_epoch
        })
        
        print(f"✅ {num_layers} layers: Test Acc = {test_acc:.4f}, "
              f"Gap = {train_acc - test_acc:.4f}, Time = {training_time:.1f}s")
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
    
    return pd.DataFrame(results)


def run_width_experiment(X_train, y_train, X_test, y_test,
                         layer_sizes: list = [50, 100, 150, 200],
                         epochs: int = 15,
                         batch_size: int = 128) -> pd.DataFrame:
    """
    Investigate effect of layer width on performance.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    layer_sizes : list
        Hidden layer sizes to test
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
        
    Returns
    -------
    pd.DataFrame
        Experiment results
    """
    print("=" * 60)
    print("WIDTH ANALYSIS EXPERIMENT")
    print(f"Testing {len(layer_sizes)} configurations")
    print("=" * 60)
    
    results = []
    early_stopping = get_early_stopping(patience=3)
    
    for layer_size in layer_sizes:
        print(f"\n🧪 Testing {layer_size}-neuron single layer MLP...")
        
        # Create model
        model = create_width_experiment_model(layer_size)
        total_params = model.count_params()
        
        # Train
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0
        )
        training_time = time.time() - start_time
        converged_epoch = len(history.history['accuracy'])
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        results.append({
            'layer_size': layer_size,
            'total_params': total_params,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'generalization_gap': train_acc - test_acc,
            'training_time': training_time,
            'converged_epoch': converged_epoch
        })
        
        print(f"✅ {layer_size} neurons: Test Acc = {test_acc:.4f}, "
              f"Gap = {train_acc - test_acc:.4f}, Time = {training_time:.1f}s")
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
    
    return pd.DataFrame(results)


def run_double_descent_experiment(X_train, y_train, X_test, y_test,
                                   hidden_sizes: list = None) -> pd.DataFrame:
    """
    Investigate double descent phenomenon with two-layer networks.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    hidden_sizes : list of tuples
        (hidden_size_1, hidden_size_2) configurations to test
        
    Returns
    -------
    pd.DataFrame
        Experiment results
    """
    if hidden_sizes is None:
        hidden_sizes = [
            # Underparameterized regime
            (20, 20), (30, 30), (50, 50), (75, 75),
            # Transition regime
            (100, 100), (150, 150), (200, 200), (300, 300),
            # Overparameterized regime
            (500, 500), (750, 750), (1000, 1000),
            (1500, 1500), (2000, 2000), (3000, 3000), (5000, 5000)
        ]
    
    print("=" * 60)
    print("DOUBLE DESCENT INVESTIGATION")
    print(f"Testing {len(hidden_sizes)} network configurations")
    print("=" * 60)
    
    results = []
    
    for i, (h1, h2) in enumerate(hidden_sizes):
        print(f"\n🧪 Configuration {i+1}/{len(hidden_sizes)}: ({h1}, {h2}) neurons")
        
        # Create model
        model = create_double_descent_model(h1, h2)
        total_params = model.count_params()
        
        # Adaptive training parameters
        if total_params < 100000:
            epochs, batch_size = 20, 128
        elif total_params < 1000000:
            epochs, batch_size = 15, 256
        else:
            epochs, batch_size = 12, 512
        
        # Train
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate risk ratio (test error / train error)
        risk_ratio = (1 - test_acc) / (1 - train_acc + 1e-10)
        
        results.append({
            'hidden_size_1': h1,
            'hidden_size_2': h2,
            'total_params': total_params,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'generalization_gap': train_acc - test_acc,
            'risk_ratio': risk_ratio,
            'training_time': training_time
        })
        
        print(f"✅ Test Acc = {test_acc:.4f}, Params = {total_params:,}, "
              f"Risk Ratio = {risk_ratio:.3f}")
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
    
    return pd.DataFrame(results)


def analyze_double_descent_curve(results: pd.DataFrame) -> dict:
    """
    Analyze the double descent curve from experiment results.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results from double descent experiment
        
    Returns
    -------
    dict
        Analysis findings
    """
    analysis = {}
    
    # Find peak performances
    best_idx = results['test_accuracy'].idxmax()
    analysis['best_config'] = {
        'hidden_sizes': (results.loc[best_idx, 'hidden_size_1'], 
                        results.loc[best_idx, 'hidden_size_2']),
        'test_accuracy': results.loc[best_idx, 'test_accuracy'],
        'params': results.loc[best_idx, 'total_params']
    }
    
    # Find the dip (if exists) - local minimum after initial improvement
    test_acc = results['test_accuracy'].values
    params = results['total_params'].values
    
    # Look for local minima
    local_mins = []
    for i in range(1, len(test_acc) - 1):
        if test_acc[i] < test_acc[i-1] and test_acc[i] < test_acc[i+1]:
            local_mins.append(i)
    
    if local_mins:
        analysis['dip_detected'] = True
        analysis['dip_configs'] = [
            {
                'hidden_sizes': (results.loc[idx, 'hidden_size_1'],
                               results.loc[idx, 'hidden_size_2']),
                'test_accuracy': results.loc[idx, 'test_accuracy'],
                'params': results.loc[idx, 'total_params']
            }
            for idx in local_mins
        ]
    else:
        analysis['dip_detected'] = False
    
    # Regime classification
    n_configs = len(results)
    analysis['regimes'] = {
        'underparameterized': results.iloc[:n_configs//3]['test_accuracy'].mean(),
        'transition': results.iloc[n_configs//3:2*n_configs//3]['test_accuracy'].mean(),
        'overparameterized': results.iloc[2*n_configs//3:]['test_accuracy'].mean()
    }
    
    return analysis


def print_experiment_summary(depth_results: pd.DataFrame = None,
                             width_results: pd.DataFrame = None,
                             dd_results: pd.DataFrame = None):
    """
    Print comprehensive summary of all experiments.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    if depth_results is not None:
        best_depth = depth_results.loc[depth_results['test_accuracy'].idxmax()]
        print(f"\n📊 DEPTH ANALYSIS:")
        print(f"   Best: {int(best_depth['num_layers'])} layers")
        print(f"   Test Accuracy: {best_depth['test_accuracy']:.4f}")
        print(f"   Parameters: {int(best_depth['total_params']):,}")
    
    if width_results is not None:
        best_width = width_results.loc[width_results['test_accuracy'].idxmax()]
        print(f"\n📊 WIDTH ANALYSIS:")
        print(f"   Best: {int(best_width['layer_size'])} neurons")
        print(f"   Test Accuracy: {best_width['test_accuracy']:.4f}")
        print(f"   Parameters: {int(best_width['total_params']):,}")
    
    if dd_results is not None:
        best_dd = dd_results.loc[dd_results['test_accuracy'].idxmax()]
        print(f"\n📊 DOUBLE DESCENT:")
        print(f"   Best: ({int(best_dd['hidden_size_1'])}, {int(best_dd['hidden_size_2'])}) neurons")
        print(f"   Test Accuracy: {best_dd['test_accuracy']:.4f}")
        print(f"   Parameters: {int(best_dd['total_params']):,}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    from .data_loader import load_mnist
    
    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist()
    
    # Run experiments (smaller scale for testing)
    depth_results = run_depth_experiment(
        X_train, y_train, X_test, y_test,
        layer_counts=[2, 4],
        epochs=5
    )
    
    width_results = run_width_experiment(
        X_train, y_train, X_test, y_test,
        layer_sizes=[50, 100],
        epochs=5
    )
    
    print_experiment_summary(depth_results, width_results)
