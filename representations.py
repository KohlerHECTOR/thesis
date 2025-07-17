# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier, MLPRegressor

zeta = 0.5
gamma= 0.99
v_opt = (zeta + gamma) / (1-gamma*gamma)
# Original dataset
x = [[0, 1, 0, 1], [0, 1, 0, 0.5], 
     [0, 1, 0.5, 1], [0.5, 1, 0, 1], 
     [0, 0.5, 0, 1], [0.5, 1, 0.5, 1], [0.5, 1, 0, 0.5], [0, 0.5, 0.5, 1], [0, 0.5, 0, 0.5]]
# y = [2, 2, 2, 1, 0, 1, 1, 0, 0]
y_reg = [
    [0.5 + gamma * v_opt, 0.5 + gamma * v_opt, v_opt, zeta + gamma * v_opt],
    [0.5 + gamma * v_opt, 0.5 + gamma * v_opt, v_opt, zeta + gamma * v_opt],
    [0.5 + gamma * v_opt, 0.5 + gamma * v_opt, v_opt, zeta + gamma * v_opt],
    [0 + gamma * v_opt, 1 + gamma * v_opt, v_opt, v_opt],
    [1 + gamma * v_opt, 0 + gamma * v_opt, v_opt, v_opt],
    [0 + gamma * v_opt, 1 + gamma * v_opt, v_opt, v_opt],
    [0 + gamma * v_opt, 1 + gamma * v_opt, v_opt, v_opt],
    [1 + gamma * v_opt, 0 + gamma * v_opt, v_opt, v_opt],
    [1 + gamma * v_opt, 0 + gamma * v_opt, v_opt, v_opt]
]

# Stack the dataset 100 times
# x = x_original * 1
# y_reg = y_reg_original * 1

# Normalize the targets
# y_reg = np.array(y_reg)
# y_reg_min = np.min(y_reg)
# y_reg_max = np.max(y_reg)
# y_reg = (y_reg - y_reg_min) / (y_reg_max - y_reg_min)

# print("Original y_reg range:", y_reg_min, "to", y_reg_max)
# print("Normalized y_reg:")
# print(y_reg)


# x = [[0, 1, 0, 1], [0, 1, 0, 0.5], 
#      [0, 1, 0.5, 1], [0.5, 1, 0, 1], 
#      [0, 0.5, 0, 1], [0.5, 1, 0.5, 1], [0.5, 1, 0, 0.5], [0, 0.5, 0.5, 1], [0, 0.5, 0, 0.5],
#      [0, 1, 0, 1], [0, 1, 0, 0.5], 
#      [0, 1, 0.5, 1], [0.5, 1, 0, 1], 
#      [0, 0.5, 0, 1], [0.5, 1, 0.5, 1], [0.5, 1, 0, 0.5], [0, 0.5, 0.5, 1], [0, 0.5, 0, 0.5]]
# y = [2, 2, 2, 3, 3, 0, 1, 1, 0, 3, 2, 2, 3, 3, 0, 1, 1, 0]
# x = [[0, 1, 0, 1], [0, 1, 0, 0.5], 
#      [0, 1, 0.5, 1], [0.5, 1, 0, 1], 
#      [0, 0.5, 0, 1], [0.5, 1, 0.5, 1], [0.5, 1, 0, 0.5], [0, 0.5, 0.5, 1], [0, 0.5, 0, 0.5],
#      ]
# y = [2, 2, 2, 3, 3, 0, 1, 1, 0]
# y_reg = [[]]

# different learning rate schedules and momentum parameters
params = [
    {
        "solver": "adam",
        "learning_rate_init": 0.05,
        "hidden_layer_sizes": (32,32,),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "learning_rate_init": 0.001,
        "hidden_layer_sizes": (32,32,),
        "activation": "relu",
    },{
        "solver": "adam",
        "learning_rate_init": 0.0001,
        "hidden_layer_sizes": (32,32,),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "learning_rate_init": 0.00001,
        "hidden_layer_sizes": (32,32,),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "learning_rate_init": 0.05,
        "hidden_layer_sizes": (32,32,),
        "activation": "tanh",
    },
    {
        "solver": "adam",
        "learning_rate_init": 0.001,
        "hidden_layer_sizes": (32,32,),
        "activation": "tanh",
    },{
        "solver": "adam",
        "learning_rate_init": 0.0001,
        "hidden_layer_sizes": (32,32,),
        "activation": "tanh",
    },
    {
        "solver": "adam",
        "learning_rate_init": 0.00001,
        "hidden_layer_sizes": (32,32,),
        "activation": "tanh",
    },
]

labels = [
    "Adam lr=5e-3 relu",
    "Adam lr=1e-3 relu",
    "Adam lr=1e-4 relu",
    "Adam lr=1e-5 relu",
    "Adam lr=5e-3 tanh",
    "Adam lr=1e-3 tanh",
    "Adam lr=1e-4 tanh",
    "Adam lr=1e-5 tanh",

]

plot_args = [
    {"c": "purple", "linestyle": "-"},
    {"c": "darkblue", "linestyle": "-"},
    {"c": "blue", "linestyle": "-"},
    {"c": "lightblue", "linestyle": "-"},
    {"c": "purple", "linestyle": "--"},
    {"c": "darkblue", "linestyle": "--"},
    {"c": "blue", "linestyle": "--"},
    {"c": "lightblue", "linestyle": "--"},
]


def plot_on_dataset(X, y, ax, name, n_runs=100):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)

    max_iter = 500
    
    # Store all loss curves for each parameter set
    all_loss_curves = {label: [] for label in labels}
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        for label, param in zip(labels, params):
            mlp = MLPRegressor(random_state=run, max_iter=max_iter, tol=1, n_iter_no_change=500, **param)
            mlp.fit(X, y)
            all_loss_curves[label].append(mlp.loss_curve_)
    
    # Plot aggregated loss curves
    for label, loss_curves in all_loss_curves.items():
        # Convert to numpy array for easier computation
        loss_curves = np.array(loss_curves)
        
        # Calculate mean and standard deviation
        mean_loss = np.mean(loss_curves, axis=0)
        # std_loss = np.std(loss_curves, axis=0)
        
        # Get the plot arguments for this label
        args = plot_args[labels.index(label)]
        
        # Plot mean curve
        iterations = range(len(mean_loss))
        ax.plot(iterations, mean_loss, label=label, linewidth=2.5, **args)
        
        # # Plot confidence interval (mean ± std)
        # ax.fill_between(iterations, 
        #                mean_loss - std_loss, 
        #                mean_loss + std_loss, 
        #                alpha=0.2, 
        #                color=args['c'])
    
    return all_loss_curves


def plot_decision_surface(mlp, ax, title):
    """Plot decision surface for x[2] = 0 and x[3] = 0.5, varying x[0] and x[1]"""
    # Create a 2D grid for x[0] and x[1]
    x0 = np.linspace(0, 1, 100)
    x1 = np.linspace(0, 1, 100)
    X0, X1 = np.meshgrid(x0, x1)
    
    # Create the full 4D input with fixed x[2] = 0 and x[3] = 0.5
    X_grid = np.zeros((X0.size, 4))
    X_grid[:, 0] = X0.ravel()  # x[0]
    X_grid[:, 1] = X1.ravel()  # x[1]
    X_grid[:, 2] = 0           # x[2] = 0 (fixed)
    X_grid[:, 3] = 1         # x[3] = 0.5 (fixed)
    
    # Get predictions
    predictions = mlp.predict(X_grid)
    predictions = predictions.reshape(X0.shape)
    
    # Plot decision surface
    contour = ax.contourf(X0, X1, predictions, levels=3, alpha=0.8, cmap='viridis')
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(contour, ax=ax, label='Predicted Class')


# Train the models
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
all_loss_curves = plot_on_dataset(x, y_reg, ax=ax1, name='test')
ax1.set_xlabel('Iteration', fontdict={'fontsize':18})
ax1.set_ylabel('Loss', fontdict={'fontsize':18})
ax1.tick_params(axis='y', labelsize=14)
ax1.legend(loc='lower left', fontsize=17)
ax1.grid(True, alpha=0.3)
# ax1.set_ylim(0, 3000)
ax1.set_yscale('log')
plt.tight_layout()
plt.savefig('repre_adam.pdf')
plt.show()