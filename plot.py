import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Configure matplotlib for PGF output
def setup_pgf():
    """Configure matplotlib to use PGF backend for LaTeX compatibility"""
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

# f = lambda gamma: (3/(1-gamma**2)-3-gamma**2-((2*gamma**2+gamma**4)/(1-gamma**2)))
# plt.show()

def compute_zeta_bounds(gamma):
    """
    Compute bounds for zeta parameter.
    
    Simplified from the original complex algebraic expressions.
    """
    
    # Common terms used in both bounds
    g = gamma
    g2 = gamma**2
    g3 = gamma**3
    g4 = gamma**4
    g5 = gamma**5
    g6 = gamma**6
    g7 = gamma**7
    
    # Lower bound - simplified expression
    # Original: ((-g4 - 2*g2)/(g3-g5) + g/(1-g)) * (g2*(1-g2)/(3*g2*(1-g2) + g2 + g5 - g7 + 2))
    
    # First part: (-g4 - 2*g2)/(g3-g5) + g/(1-g)
    # = -g2(g2 + 2)/(g3(1-g2)) + g/(1-g)
    # Combined: [2*(g-1) + g3*(1-g)] / [g*(1-g2)*(1-g)]
    
    lower_numerator = 2*(g - 1) + g3*(1 - g)
    lower_denom1 = g * (1 - g2) * (1 - g)
    
    # Second part denominator: 4*g2 - 3*g4 + g5 - g7 + 2
    lower_denom2 = 4*g2 - 3*g4 + g5 - g7 + 2
    
    # Complete lower bound
    lower = (lower_numerator * g2 * (1 - g2)) / (lower_denom1 * lower_denom2)
    
    # Upper bound - simplified expression  
    # Original: ((g3+g5-g4-g7)/(1-g2)) * (1/(2*g+g3+g4-g2-((g2+g4-g3-g6)/(1-g2))))
    
    upper_numerator = g3 + g5 - g4 - g7
    
    # Inner fraction: (g2+g4-g3-g6)/(1-g2) = g2/(1+g) + g4
    inner_fraction = g2/(1 + g) + g4
    
    # Complete denominator
    upper_denominator = (1 - g2) * (2*g + g3 + g4 - g2 - inner_fraction)
    
    upper = upper_numerator / upper_denominator
    
    return lower, upper

def plot_bounds(gamma_values=None):
    """Plot the bounds for a range of gamma values"""
    if gamma_values is None:
        gamma_values = np.linspace(0.0001, 0.9999, 9999)
    
    lower_bounds = []
    upper_bounds = []
    valid_regions = []
    
    for gamma in gamma_values:
        lower, upper = compute_zeta_bounds(gamma)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        valid_regions.append(upper > lower)
    
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_values, lower_bounds, 'b-', label='Lower bound')
    plt.plot(gamma_values, upper_bounds, 'r-', label='Upper bound')
    
    # Shade the valid region
    valid_gamma = [gamma_values[i] for i in range(len(gamma_values)) if valid_regions[i]]
    valid_lower = [lower_bounds[i] for i in range(len(gamma_values)) if valid_regions[i]]
    valid_upper = [upper_bounds[i] for i in range(len(gamma_values)) if valid_regions[i]]
    
    if valid_gamma:
        plt.fill_between(valid_gamma, valid_lower, valid_upper, color='green', alpha=0.3, 
                         label='Valid region')
    
    plt.xlabel('γ (gamma)')
    plt.ylabel('ζ (zeta)')
    plt.title('Bounds on ζ where J(T1) > J(T0) and J(T1) > J(T2)')
    plt.legend()
    plt.grid(True)
    plt.savefig('zeta_bounds.png')
    plt.show()

def plot_bounds_pgf(gamma_values=None, output_file="zeta_bounds"):
    """Generate PGF version of the bounds plot for LaTeX documents"""
    setup_pgf()
    
    if gamma_values is None:
        gamma_values = np.linspace(0.0001, 0.9999, 1000)  # Reduced points for cleaner PGF
    
    lower_bounds = []
    upper_bounds = []
    valid_regions = []
    
    for gamma in gamma_values:
        lower, upper = compute_zeta_bounds(gamma)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        valid_regions.append(upper > lower)
    
    plt.figure(figsize=(5, 3.5))  # Smaller size for papers
    plt.plot(gamma_values, lower_bounds, 'b-', linewidth=1.2, label='Lower bound')
    plt.plot(gamma_values, upper_bounds, 'r-', linewidth=1.2, label='Upper bound')
    
    # Shade the valid region
    valid_gamma = [gamma_values[i] for i in range(len(gamma_values)) if valid_regions[i]]
    valid_lower = [lower_bounds[i] for i in range(len(gamma_values)) if valid_regions[i]]
    valid_upper = [upper_bounds[i] for i in range(len(gamma_values)) if valid_regions[i]]
    
    if valid_gamma:
        plt.fill_between(valid_gamma, valid_lower, valid_upper, color='green', alpha=0.2, 
                         label='Valid region')
    
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$\zeta$')
    plt.title(r'Bounds on $\zeta$ where $J(T_1) > J(T_0)$ and $J(T_1) > J(T_2)$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save as PGF
    plt.savefig(f'{output_file}.pgf')
    plt.savefig(f'{output_file}.pdf')  # Also save PDF for preview
    print(f"PGF plot saved as {output_file}.pgf")
    print(f"PDF preview saved as {output_file}.pdf")

def export_data_for_pgfplots(gamma_values=None, output_file="zeta_bounds_data.dat"):
    """Export data in format suitable for direct use with pgfplots"""
    if gamma_values is None:
        gamma_values = np.linspace(0.0001, 0.9999, 200)  # Even fewer points for manual plotting
    
    with open(output_file, 'w') as f:
        # Column headers without # comment (pgfplots compatible)
        f.write("gamma lower_bound upper_bound valid\n")
        for gamma in gamma_values:
            lower, upper = compute_zeta_bounds(gamma)
            valid = 1 if upper > lower else 0
            f.write(f"{gamma:.6f} {lower:.6f} {upper:.6f} {valid}\n")
    
    print(f"Data exported to {output_file}")
    print("Use this data with pgfplots like:")
    print("""
\\begin{tikzpicture}
\\begin{axis}[
    xlabel={$\\gamma$},
    ylabel={$\\zeta$},
    legend pos=north west,
    grid=major,
]
\\addplot[blue, thick] table[x=gamma, y=lower_bound] {""" + output_file + """};
\\addplot[red, thick] table[x=gamma, y=upper_bound] {""" + output_file + """};
\\legend{Lower bound, Upper bound}
\\end{axis}
\\end{tikzpicture}
    """)

def generate_pgfplots_code(gamma_values=None, output_file="zeta_bounds_plot.tex"):
    """Generate complete pgfplots LaTeX code"""
    if gamma_values is None:
        gamma_values = np.linspace(0.0001, 0.9999, 100)
    
    # Calculate bounds
    data_points = []
    for gamma in gamma_values:
        lower, upper = compute_zeta_bounds(gamma)
        if upper > lower:  # Only include valid regions
            data_points.append((gamma, lower, upper))
    
    with open(output_file, 'w') as f:
        f.write("""\\documentclass{standalone}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.17}

\\begin{document}
\\begin{tikzpicture}
\\begin{axis}[
    width=10cm,
    height=6cm,
    xlabel={$\\gamma$},
    ylabel={$\\zeta$},
    title={Bounds on $\\zeta$ where $J(T_1) > J(T_0)$ and $J(T_1) > J(T_2)$},
    legend pos=north west,
    grid=major,
    grid style={gray!30},
]

% Lower bound
\\addplot[blue, thick, smooth] coordinates {
""")
        for gamma, lower, upper in data_points:
            f.write(f"    ({gamma:.4f}, {lower:.4f})\n")
        
        f.write("""};

% Upper bound  
\\addplot[red, thick, smooth] coordinates {
""")
        for gamma, lower, upper in data_points:
            f.write(f"    ({gamma:.4f}, {upper:.4f})\n")
            
        f.write("""};

% Fill between (valid region)
\\addplot[green!20, forget plot] fill between[of=A and B];

\\legend{Lower bound, Upper bound}
\\end{axis}
\\end{tikzpicture}
\\end{document}
""")
    
    print(f"Complete pgfplots LaTeX code saved to {output_file}")
    print("Compile with: pdflatex " + output_file)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "pgf":
        # Generate PGF version
        print("Generating PGF version...")
        plot_bounds_pgf()
    elif len(sys.argv) > 1 and sys.argv[1] == "data":
        # Export data for pgfplots
        print("Exporting data for pgfplots...")
        export_data_for_pgfplots()
    elif len(sys.argv) > 1 and sys.argv[1] == "tex":
        # Generate complete LaTeX file
        print("Generating complete pgfplots LaTeX code...")
        generate_pgfplots_code()
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        # Generate all formats
        print("Generating all formats...")
        plot_bounds()  # Standard matplotlib plot
        plot_bounds_pgf()  # PGF version
        export_data_for_pgfplots()  # Data file
        generate_pgfplots_code()  # Complete LaTeX code
    else:
        # Default: standard matplotlib plot
        plot_bounds()