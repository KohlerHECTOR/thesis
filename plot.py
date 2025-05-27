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
    Compute bounds for zeta parameter based on the derived inequalities.
    
    From the three inequalities, we derived:
    - Lower bound: ζ ≥ (γ² + γ + 1 - 2γ³ - γ⁵)/4
    - Upper bound 1: ζ ≤ (γ² + γ⁴)/2  
    - Upper bound 2: ζ ≤ (1 + 2γ² + γ⁴)/4
    
    The effective upper bound is the minimum of the two upper bounds.
    """
    
    g = gamma
    g2 = gamma**2
    g3 = gamma**3
    g4 = gamma**4
    g5 = gamma**5
    
    # Lower bound: (γ² + γ + 1 - 2γ³ - γ⁵)/4
    lower = (g2 + g + 1 - 2*g3 - g5) / 4
    
    # Upper bound 1: (γ² + γ⁴)/2
    upper1 = (g2 + g4) / 2
    
    # Upper bound 2: (1 + 2γ² + γ⁴)/4
    upper2 = (1 + 2*g2 + g4) / 4
    
    # Effective upper bound is the minimum of the two
    upper = min(upper1, upper2)
    
    return lower, upper

def plot_bounds(gamma_values=None, show_individual_bounds=False):
    """Plot the bounds for a range of gamma values"""
    if gamma_values is None:
        gamma_values = np.linspace(0.0001, 0.9999, 9999)
    
    lower_bounds = []
    upper_bounds = []
    upper1_bounds = []
    upper2_bounds = []
    valid_regions = []
    
    for gamma in gamma_values:
        g = gamma
        g2 = gamma**2
        g3 = gamma**3
        g4 = gamma**4
        g5 = gamma**5
        
        # Calculate all bounds
        lower = (g2 + g + 1 - 2*g3 - g5) / 4
        upper1 = (g2 + g4) / 2
        upper2 = (1 + 2*g2 + g4) / 4
        upper = min(upper1, upper2)
        
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        upper1_bounds.append(upper1)
        upper2_bounds.append(upper2)
        valid_regions.append(upper > lower)
    
    plt.figure(figsize=(12, 8))
    plt.plot(gamma_values, lower_bounds, 'b-', linewidth=2, label='Lower bound: (γ² + γ + 1 - 2γ³ - γ⁵)/4')
    
    if show_individual_bounds:
        plt.plot(gamma_values, upper1_bounds, 'r--', linewidth=1.5, label='Upper bound 1: (γ² + γ⁴)/2')
        plt.plot(gamma_values, upper2_bounds, 'm--', linewidth=1.5, label='Upper bound 2: (1 + 2γ² + γ⁴)/4')
        plt.plot(gamma_values, upper_bounds, 'r-', linewidth=2, label='Effective upper bound: min(bound1, bound2)')
    else:
        plt.plot(gamma_values, upper_bounds, 'r-', linewidth=2, label='Upper bound')
    
    # Shade the valid region
    valid_gamma = [gamma_values[i] for i in range(len(gamma_values)) if valid_regions[i]]
    valid_lower = [lower_bounds[i] for i in range(len(gamma_values)) if valid_regions[i]]
    valid_upper = [upper_bounds[i] for i in range(len(gamma_values)) if valid_regions[i]]
    
    if valid_gamma:
        plt.fill_between(valid_gamma, valid_lower, valid_upper, color='green', alpha=0.2, 
                         label='Feasible region for ζ')
    
    plt.xlabel('γ (gamma)')
    plt.ylabel('ζ (zeta)')
    plt.title('Bounds on ζ from derived inequalities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('zeta_bounds.png', dpi=300, bbox_inches='tight')
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
    plt.title(r'Bounds on $\zeta$ from derived inequalities')
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
    title={Bounds on $\\zeta$ from derived inequalities},
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
    elif len(sys.argv) > 1 and sys.argv[1] == "detailed":
        # Show individual bounds
        print("Plotting with individual bounds shown...")
        plot_bounds(show_individual_bounds=True)
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