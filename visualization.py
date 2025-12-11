"""
Visualization Module for MNIST Neural Network Analysis
=======================================================
Creates plots and visualizations for experiment results.

Author: Victor Prefa
Course: SIG720 Machine Learning, Deakin University
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_depth_analysis(results: pd.DataFrame, save_path: str = None):
    """
    Create comprehensive depth analysis visualization.
    
    Parameters
    ----------
    results : pd.DataFrame
        Depth experiment results
    save_path : str, optional
        Path to save figure
    """
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MNIST MLP Depth Analysis', fontsize=16, fontweight='bold')
    
    num_layers = results['num_layers'].values
    
    # 1. Test Accuracy
    axes[0, 0].plot(num_layers, results['test_accuracy'] * 100, 'bo-', 
                    linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Hidden Layers')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].set_title('Test Accuracy vs Depth')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Generalization Gap
    axes[0, 1].plot(num_layers, results['generalization_gap'] * 100, 'ro-',
                    linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Hidden Layers')
    axes[0, 1].set_ylabel('Generalization Gap (%)')
    axes[0, 1].set_title('Overfitting vs Depth')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Parameters
    axes[0, 2].plot(num_layers, results['total_params'] / 1000, 'go-',
                    linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Number of Hidden Layers')
    axes[0, 2].set_ylabel('Parameters (Thousands)')
    axes[0, 2].set_title('Model Complexity vs Depth')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Training Time
    axes[1, 0].plot(num_layers, results['training_time'], 'mo-',
                    linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Hidden Layers')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time vs Depth')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Train vs Test
    axes[1, 1].plot(num_layers, results['train_accuracy'] * 100, 'b^-',
                    label='Training', linewidth=2, markersize=8)
    axes[1, 1].plot(num_layers, results['test_accuracy'] * 100, 'ro-',
                    label='Test', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Hidden Layers')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Training vs Test Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Convergence
    axes[1, 2].plot(num_layers, results['converged_epoch'], 'co-',
                    linewidth=2, markersize=8)
    axes[1, 2].set_xlabel('Number of Hidden Layers')
    axes[1, 2].set_ylabel('Epochs to Convergence')
    axes[1, 2].set_title('Convergence Speed vs Depth')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_width_analysis(results: pd.DataFrame, save_path: str = None):
    """
    Create width analysis visualization.
    
    Parameters
    ----------
    results : pd.DataFrame
        Width experiment results
    save_path : str, optional
        Path to save figure
    """
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MNIST MLP Width Analysis', fontsize=16, fontweight='bold')
    
    layer_size = results['layer_size'].values
    
    # 1. Test Accuracy
    axes[0, 0].plot(layer_size, results['test_accuracy'] * 100, 'bo-',
                    linewidth=2, markersize=10)
    axes[0, 0].set_xlabel('Number of Neurons')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].set_title('Test Accuracy vs Width')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Parameters vs Accuracy
    axes[0, 1].scatter(results['total_params'] / 1000, results['test_accuracy'] * 100,
                      s=100, c=layer_size, cmap='viridis')
    axes[0, 1].set_xlabel('Parameters (Thousands)')
    axes[0, 1].set_ylabel('Test Accuracy (%)')
    axes[0, 1].set_title('Accuracy vs Model Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Generalization Gap
    axes[1, 0].bar(layer_size.astype(str), results['generalization_gap'] * 100,
                   color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('Number of Neurons')
    axes[1, 0].set_ylabel('Generalization Gap (%)')
    axes[1, 0].set_title('Overfitting vs Width')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Training Time
    axes[1, 1].plot(layer_size, results['training_time'], 'go-',
                    linewidth=2, markersize=10)
    axes[1, 1].set_xlabel('Number of Neurons')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_title('Training Time vs Width')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_double_descent(results: pd.DataFrame, save_path: str = None):
    """
    Create double descent curve visualization.
    
    Parameters
    ----------
    results : pd.DataFrame
        Double descent experiment results
    save_path : str, optional
        Path to save figure
    """
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MNIST Double Descent Investigation', fontsize=16, fontweight='bold')
    
    params = results['total_params'].values / 1000  # Convert to thousands
    
    # 1. Test Accuracy vs Parameters (THE DOUBLE DESCENT CURVE)
    axes[0, 0].plot(params, results['test_accuracy'] * 100, 'bo-',
                    linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Parameters (Thousands)')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].set_title('Double Descent: Test Accuracy vs Model Size')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=100, color='r', linestyle='--', alpha=0.5, label='Interpolation threshold')
    axes[0, 0].legend()
    
    # 2. Test Error (Risk) vs Parameters
    test_error = (1 - results['test_accuracy']) * 100
    axes[0, 1].plot(params, test_error, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Parameters (Thousands)')
    axes[0, 1].set_ylabel('Test Error (%)')
    axes[0, 1].set_title('Double Descent: Test Error (Risk Curve)')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Generalization Gap
    axes[1, 0].plot(params, results['generalization_gap'] * 100, 'go-',
                    linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Parameters (Thousands)')
    axes[1, 0].set_ylabel('Generalization Gap (%)')
    axes[1, 0].set_title('Generalization Gap vs Model Size')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Train vs Test
    axes[1, 1].plot(params, results['train_accuracy'] * 100, 'b^-',
                    label='Training', linewidth=2, markersize=8)
    axes[1, 1].plot(params, results['test_accuracy'] * 100, 'ro-',
                    label='Test', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Parameters (Thousands)')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Training vs Test Accuracy')
    axes[1, 1].set_xscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_training_history(history, title: str = "Training History"):
    """
    Plot training history curves.
    
    Parameters
    ----------
    history : keras History object
        Training history
    title : str
        Plot title
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_comparison_table(depth_results: pd.DataFrame,
                            width_results: pd.DataFrame) -> pd.DataFrame:
    """
    Create comparison table of depth vs width experiments.
    
    Parameters
    ----------
    depth_results : pd.DataFrame
        Depth experiment results
    width_results : pd.DataFrame
        Width experiment results
        
    Returns
    -------
    pd.DataFrame
        Comparison summary
    """
    best_depth = depth_results.loc[depth_results['test_accuracy'].idxmax()]
    best_width = width_results.loc[width_results['test_accuracy'].idxmax()]
    
    comparison = pd.DataFrame({
        'Metric': ['Best Config', 'Test Accuracy', 'Parameters', 
                   'Training Time', 'Gen. Gap'],
        'Depth Approach': [
            f"{int(best_depth['num_layers'])} layers × 100",
            f"{best_depth['test_accuracy']:.4f}",
            f"{int(best_depth['total_params']):,}",
            f"{best_depth['training_time']:.1f}s",
            f"{best_depth['generalization_gap']:.4f}"
        ],
        'Width Approach': [
            f"{int(best_width['layer_size'])} neurons",
            f"{best_width['test_accuracy']:.4f}",
            f"{int(best_width['total_params']):,}",
            f"{best_width['training_time']:.1f}s",
            f"{best_width['generalization_gap']:.4f}"
        ]
    })
    
    return comparison


if __name__ == "__main__":
    # Test with sample data
    print("Visualization module loaded successfully.")
    print("Available functions:")
    print("  - plot_depth_analysis(results)")
    print("  - plot_width_analysis(results)")
    print("  - plot_double_descent(results)")
    print("  - plot_training_history(history)")
    print("  - create_comparison_table(depth_results, width_results)")
