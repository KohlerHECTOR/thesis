import numpy as np
import matplotlib.pyplot as plt

def objective_depth0(zeta, gamma):
    """Objective value for depth-0 decision tree"""
    # This doesn't depend on zeta, so return constant array
    return np.full_like(zeta, (1 + gamma) / (4 * (1 - gamma)))

def objective_depth1(zeta, gamma):
    """Objective value for depth-1 decision tree"""
    return (4*zeta + gamma + 2*gamma**3 + gamma**5) / (4 * (1 - gamma**2))

def objective_unbalanced_depth2(zeta, gamma):
    """Objective value for unbalanced depth-2 decision tree"""
    numerator = zeta*(4 + 2*gamma - 2*gamma**2 - gamma**5 + gamma**6) + gamma + gamma**3 + gamma**4 + gamma**7
    denominator = 4 * (1 - gamma**2)
    return numerator / denominator

def objective_balanced_depth2(zeta, gamma):
    """Objective value for balanced depth-2 decision tree"""
    numerator = zeta*(3 + 3*gamma) + gamma**2 + gamma**5 + gamma**8
    denominator = 4 * (1 - gamma**3)
    return numerator / denominator

def objective_infinite(zeta, gamma):
    """Objective value for infinite tree"""
    return zeta / (1 - gamma)

def objective_stochastic(zeta, gamma):
    """
    Compute the objective value for the stochastic policy.
    
    Formula: (1 + 1.5*gamma + 0.5*gamma^2) / (4*(1 - 0.25*gamma^2)*(1-gamma))
    
    Parameters:
    gamma: discount factor (scalar or array)
    
    Returns:
    Objective value(s)
    """
    numerator = 1 + 1.5 * gamma + 0.5 * gamma**2
    denominator = 4 * (1 - 0.25 * gamma**2) * (1 - gamma)
    return [numerator / denominator for z in zeta]

# Parameters
gamma = 0.99
zeta_range = np.linspace(-1, 2, 1000)

# Calculate objective values
J0 = objective_depth0(zeta_range, gamma)
J1 = objective_depth1(zeta_range, gamma)
Ju = objective_unbalanced_depth2(zeta_range, gamma)
J2 = objective_balanced_depth2(zeta_range, gamma)
Jinf = objective_infinite(zeta_range, gamma)
Jstoc = objective_stochastic(zeta_range, gamma)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each tree's objective value
ax.plot(zeta_range, J0, label='$\\pi_{\\mathcal{T}_0}$', linewidth=2, color='red')
ax.plot(zeta_range, J1, label='$\\pi_{\\mathcal{T}_1}$', linewidth=2, color='green')
ax.plot(zeta_range, Ju, label='$\\pi_{\\mathcal{T}_u}$', linewidth=2, color='orange')
ax.plot(zeta_range, J2, label='$\\pi_{\\mathcal{T}_2}$', linewidth=2, color='blue')
ax.plot(zeta_range, Jinf, label='Infinite Tree', linewidth=2,
        color='purple', linestyle='--')
ax.plot(zeta_range, Jstoc, label='Stochastic', linewidth=2,
        color='black', linestyle='--')

# Find the optimal tree for each zeta value
all_objectives = np.column_stack([J0, J1, Ju, J2, Jinf])
optimal_indices = np.argmax(all_objectives, axis=1)

tree_names = [
    '$\\pi_{\\mathcal{T}_0}$',
    '$\\pi_{\\mathcal{T}_1}$',
    '$\\pi_{\\mathcal{T}_u}$',
    '$\\pi_{\\mathcal{T}_2}$',
    'Infinite Tree'
]
colors = ['red', 'green', 'orange', 'blue', 'purple']

# Highlight the optimal regions
for i, tree_name in enumerate(tree_names):
    mask = optimal_indices == i
    if np.any(mask):
        ax.fill_between(
            zeta_range[mask],
            0,
            all_objectives[mask, i],
            alpha=0.3,
            color=colors[i],
            label=f'$\\pi^*$ = {tree_name}'
        )

# Labels, legend, and styling
ax.set_xlabel('$\\zeta$', fontsize=32)
ax.set_ylabel('$V^{\\pi}(o_0)$, $\\gamma=0.99$', fontsize=25)

ax.legend(fontsize=25, loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.grid(True, alpha=0.3)

ax.set_xlim(-1, 2)
ax.set_ylim(-5, 110)

ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)

fig.tight_layout(rect=[0, 0, 1, 1])

# Save the plot
fig.savefig('images/images_part1/objective_values_plot.pdf')