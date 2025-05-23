import numpy as np
import matplotlib.pyplot as plt

f = lambda gamma: (3/(1-gamma**2)-3-gamma**2-((2*gamma**2+gamma**4)/(1-gamma**2)))
plt.show()

# def compute_zeta_bounds(gamma):
    
#     lower = ((-gamma**4 - 2*gamma**2)/(gamma**3-gamma**5)+gamma/(1-gamma))*((gamma**2*(1-gamma**2))/(3*gamma**2*(1-gamma**2)+gamma**2+gamma**5-gamma**7+2))
#     upper = ((gamma**3+gamma**5-gamma**4-gamma**7)/(1-gamma**2))*(1/(2*gamma+gamma**3+gamma**4-gamma**2-((gamma**2+gamma**4-gamma**3-gamma**6)/(1-gamma**2))))
#     other_upper = ((2*gamma**3+gamma**5)/(1-gamma**2)) * (1/(3/(1-gamma**2)-3-gamma**2-((2*gamma**2+gamma**4)/(1-gamma**2))))
#     upper = np.minimum(upper,((2*gamma**3+gamma**5)/(1-gamma**2)) * (1/(3/(1-gamma**2)-3-gamma**2-((2*gamma**2+gamma**4)/(1-gamma**2)))))
#     return lower, upper

# def plot_bounds(gamma_values=None):
#     """Plot the bounds for a range of gamma values"""
#     if gamma_values is None:
#         gamma_values = np.linspace(0.01, 0.99, 100)
    
#     lower_bounds = []
#     upper_bounds = []
#     valid_regions = []
    
#     for gamma in gamma_values:
#         lower, upper = compute_zeta_bounds(gamma)
#         lower_bounds.append(lower)
#         upper_bounds.append(upper)
#         valid_regions.append(upper > lower)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(gamma_values, lower_bounds, 'b-', label='Lower bound')
#     plt.plot(gamma_values, upper_bounds, 'r-', label='Upper bound')
    
#     # Shade the valid region
#     valid_gamma = [gamma_values[i] for i in range(len(gamma_values)) if valid_regions[i]]
#     valid_lower = [lower_bounds[i] for i in range(len(gamma_values)) if valid_regions[i]]
#     valid_upper = [upper_bounds[i] for i in range(len(gamma_values)) if valid_regions[i]]
    
#     if valid_gamma:
#         plt.fill_between(valid_gamma, valid_lower, valid_upper, color='green', alpha=0.3, 
#                          label='Valid region')
    
#     plt.xlabel('γ (gamma)')
#     plt.ylabel('ζ (zeta)')
#     plt.title('Bounds on ζ where J(T1) > J(T0) and J(T1) > J(T2)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('zeta_bounds.png')
#     plt.show()

# if __name__ == "__main__":
#     # Example usage
#     gamma = float(input("Enter a value for gamma (between 0 and 1): "))
#     lower, upper = compute_zeta_bounds(gamma)
    
#     print(f"For γ = {gamma}:")
#     print(f"Lower bound on ζ: {lower:.6f}")
#     print(f"Upper bound on ζ: {upper:.6f}")
    
#     if lower < upper:
#         print(f"Valid range for ζ: ({lower:.6f}, {upper:.6f})")
#     else:
#         print("No valid values for ζ at this gamma value.")
    
#     # Plot bounds for a range of gamma values
#     plot_choice = input("Would you like to see a plot of the bounds for different gamma values? (y/n): ")
#     if plot_choice.lower() == 'y':
#         plot_bounds()