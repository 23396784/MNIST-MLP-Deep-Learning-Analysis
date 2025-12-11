"""
MNIST Neural Network Analysis
=============================

A comprehensive investigation of MLP architectures on MNIST digit classification,
exploring depth, width, and the double descent phenomenon.

Modules:
    - data_loader: MNIST data loading and preprocessing
    - mlp_models: MLP model architectures
    - experiments: Depth, width, and double descent experiments
    - visualization: Result plotting and visualization

Author: Victor Prefa
Course: SIG720 Machine Learning, Deakin University
"""

from .data_loader import (
    load_mnist,
    flatten_images,
    get_data_info,
    visualize_samples
)

from .mlp_models import (
    create_single_layer_mlp,
    create_mlp,
    create_depth_experiment_model,
    create_width_experiment_model,
    create_double_descent_model,
    get_early_stopping,
    count_parameters
)

from .experiments import (
    run_depth_experiment,
    run_width_experiment,
    run_double_descent_experiment,
    analyze_double_descent_curve,
    print_experiment_summary
)

from .visualization import (
    plot_depth_analysis,
    plot_width_analysis,
    plot_double_descent,
    plot_training_history,
    create_comparison_table
)

__version__ = '1.0.0'
__author__ = 'Victor Prefa'
