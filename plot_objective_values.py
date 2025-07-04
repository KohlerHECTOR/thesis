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
plt.figure(figsize=(12, 8))

# Plot each tree's objective value
plt.plot(zeta_range, J0, label='Depth-0 Tree', linewidth=2, color='blue')
plt.plot(zeta_range, J1, label='Depth-1 Tree', linewidth=2, color='red')
plt.plot(zeta_range, Ju, label='Unbalanced Depth-2 Tree', linewidth=2, color='green')
plt.plot(zeta_range, J2, label='Balanced Depth-2 Tree', linewidth=2, color='orange')
plt.plot(zeta_range, Jinf, label='Infinite Tree', linewidth=2, color='purple', linestyle='--')
plt.plot(zeta_range, Jstoc, label='Stochastic Depth-0 Tree', linewidth=2, color='black', linestyle='--')

# Find the optimal tree for each zeta value
all_objectives = np.column_stack([J0, J1, Ju, J2, Jinf])
optimal_indices = np.argmax(all_objectives, axis=1)
tree_names = ['Depth-0', 'Depth-1', 'Unbalanced Depth-2', 'Balanced Depth-2', 'Infinite']
optimal_tree = [tree_names[i] for i in optimal_indices]

# Highlight the optimal regions
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, tree_name in enumerate(tree_names):
    mask = optimal_indices == i
    if np.any(mask):
        plt.fill_between(zeta_range[mask], 0, all_objectives[mask, i], 
                        alpha=0.2, color=colors[i], label=f'{tree_name} (optimal)')

plt.xlabel(r'$\zeta$ (interpretability penalty)', fontsize=14)
plt.ylabel('$M_{IB}$ Value, $\gamma=0.99$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-1, 2)

# Add some analysis
print("Analysis of optimal trees:")
print("=" * 50)
for i, tree_name in enumerate(tree_names):
    count = np.sum(optimal_indices == i)
    percentage = count / len(zeta_range) * 100
    print(f"{tree_name}: optimal for {count} values ({percentage:.1f}% of ζ range)")

# Find transition points
transitions = []
for i in range(1, len(optimal_indices)):
    if optimal_indices[i] != optimal_indices[i-1]:
        transitions.append((zeta_range[i], tree_names[optimal_indices[i-1]], tree_names[optimal_indices[i]]))

print(f"\nTransition points:")
for zeta, from_tree, to_tree in transitions:
    print(f"At ζ = {zeta:.3f}: {from_tree} → {to_tree}")

plt.tight_layout()

# Save the plot
plt.savefig('images/images_part1/objective_values_plot.pdf')