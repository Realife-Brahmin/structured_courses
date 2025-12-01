#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Plots

# Pine Rose (Rosé Pine) theme - matching your VSCode aesthetic
rose_base = "#191724"      # Deep purple-black background
rose_surface = "#1f1d2e"   # Slightly lighter surface
rose_overlay = "#26233a"   # Overlay/subtle elements
rose_muted = "#6e6a86"     # Muted/dimmed text
rose_text = "#e0def4"      # Main text color
rose_love = "#eb6f92"      # Love/pink-red
rose_gold = "#f6c177"      # Gold/orange
rose_rose = "#ebbcba"      # Rose/warm pink
rose_pine = "#31748f"      # Pine/teal-blue
rose_foam = "#9ccfd8"      # Foam/cyan
rose_iris = "#c4a7e7"      # Iris/purple

println("\n" * "="^80)
println("Problem 2 - Creating Dao-Themed Trajectory Plot")
println("="^80)

# Define the ODE: (e^t + 1)x' + xe^t - x = 0
# Rewritten as: x' = x(1 - e^t)/(e^t + 1)
f(t, x) = x * (1 - exp(t)) / (exp(t) + 1)

# Exact solution
exact_solution(t) = 12 * exp(t) / (exp(t) + 1)^2

# RK4 implementation
function rk4_step(f, t, x, h)
    k1 = f(t, x)
    k2 = f(t + h/2, x + h*k1/2)
    k3 = f(t + h/2, x + h*k2/2)
    k4 = f(t + h, x + h*k3)
    return x + h * (k1 + 2*k2 + 2*k3 + k4) / 6
end

function rk4_solve(f, t0, tf, x0, h)
    n_steps = abs(Int((tf - t0) / h))
    t_values = zeros(n_steps + 1)
    x_values = zeros(n_steps + 1)
    
    t_values[1] = t0
    x_values[1] = x0
    
    t = t0
    x = x0
    
    for i in 1:n_steps
        x = rk4_step(f, t, x, h)
        t += h
        t_values[i+1] = t
        x_values[i+1] = x
    end
    
    return t_values, x_values
end

# Solve the ODE
println("\nSolving ODE with RK4...")
t0 = 0.0
tf = -2.0
x0 = 3.0
h = -0.01

t_rk4, x_rk4 = rk4_solve(f, t0, tf, x0, h)

# Generate analytical solution for the same time points
x_exact = exact_solution.(t_rk4)

println(@sprintf("Generated %d points from t=%.2f to t=%.2f", length(t_rk4), t0, tf))

# Create the Dao-themed plot
println("\nCreating Pine Rose themed plot...")

# Create plot with Pine Rose aesthetic
plot(
    t_rk4, x_exact,
    label="Analytical Solution",
    linewidth=4,
    linestyle=:solid,
    color=rose_foam,  # Cyan/blue for analytical truth
    legend=:topright,
    legendfontcolor=rose_text,
    legendfontsize=10,
    xlabel="Time t",
    ylabel="x(t)",
    title="Problem 2: ODE Solution Comparison\n(e^t + 1)x' + xe^t - x = 0, x(0) = 3",
    titlefontsize=12,
    titlefontcolor=rose_text,
    labelfontsize=11,
    guidefontcolor=rose_text,
    tickfontcolor=rose_muted,
    background_color=rose_base,
    foreground_color=rose_text,
    grid=true,
    minorgrid=true,
    gridcolor=rose_overlay,
    minorgridcolor=rose_surface,
    gridlinewidth=1.5,
    minorgridlinewidth=0.5,
    gridalpha=0.6,
    minorgridalpha=0.3,
    framestyle=:box,
    size=(800, 600),
    dpi=300,
    margin=5Plots.mm
)

plot!(
    t_rk4, x_rk4,
    label="RK4 Numerical Solution",
    linewidth=4,
    linestyle=:dash,
    color=rose_gold  # Warm orange/gold for numerical
)

# Add a subtle annotation in theme colors
annotate!(
    -1.0, 1.5,
    text("Step size h = -0.01\n200 steps", rose_muted, 9, :left)
)

# Ensure directories exist
processedData_dir = "../processedData"
figures_dir = "../tex/figures"

if !isdir(processedData_dir)
    mkpath(processedData_dir)
    println("Created directory: $processedData_dir")
end

if !isdir(figures_dir)
    mkpath(figures_dir)
    println("Created directory: $figures_dir")
end

# Save the plot
output_file = joinpath(processedData_dir, "p02_trajectory_comparison.png")
println("\nSaving plot to: $output_file")
savefig(output_file)
println("✓ Plot saved successfully!")

# Copy to tex/figures folder
tex_output = joinpath(figures_dir, "p02_trajectory_comparison.png")
println("Copying to: $tex_output")
cp(output_file, tex_output, force=true)
println("✓ Plot copied to tex/figures/")

println("\n" * "="^80)
println("Plot generation complete!")
println("="^80 * "\n")
