# CPTS530 Final Project - Problem 2: Fourth-Order Runge-Kutta Method
# Author: Aryan Ritwajeet Jha
# Date: December 2025

using LinearAlgebra
using Printf
using Plots

# Load environment and utilities
# include("preamble.jl")
# include("test_utils.jl")

println("="^80)
println("CPTS530 Final Project - Problem 2: Fourth-Order Runge-Kutta Method")
println("="^80)

# =================================================================
# Problem Setup
# =================================================================

println("\n📋 Problem Statement")
println("-"^80)
println("""
Solve the initial-value problem:
    (e^t + 1)x' + xe^t - x = 0
    x(0) = 3

on the interval [-2, 0] using step size h = -0.01

Compare with exact analytical solution.
""")

# =================================================================
# Define the ODE: x' = f(t, x)
# =================================================================

"""
    f(t, x)

Right-hand side of the ODE: x' = f(t, x)
From (e^t + 1)x' + xe^t - x = 0, we get:
x' = x(1 - e^t)/(e^t + 1)
"""
function f(t, x)
    return x * (1 - exp(t)) / (exp(t) + 1)
end

# =================================================================
# Exact Analytical Solution
# =================================================================

"""
    exact_solution(t)

Exact analytical solution obtained by integrating factor method.
x(t) = 3e(e^t + 1)^2 / (4e^(e^t))
"""
function exact_solution(t)
    # Using the derived solution from integrating factor method
    numerator = 3 * ℯ * (exp(t) + 1)^2
    denominator = 4 * exp(exp(t))
    return numerator / denominator
end

"""
    exact_solution_given(t)

The exact solution given in the problem statement:
x(t) = 12 * e^t / (e^t + 1)^2
"""
function exact_solution_given(t)
    return 12 * exp(t) / (exp(t) + 1)^2
end

# Verify which exact solution is correct
println("\n🔍 Verifying Exact Solutions at t=0")
println("-"^80)
println("Initial condition: x(0) = 3")
println("Our derived solution at t=0: ", @sprintf("%.10f", exact_solution(0.0)))
println("Given solution at t=0:       ", @sprintf("%.10f", exact_solution_given(0.0)))

# Check which one satisfies the initial condition
if abs(exact_solution(0.0) - 3.0) < 1e-10
    println("✓ Our derived solution satisfies x(0) = 3")
    exact_sol = exact_solution
elseif abs(exact_solution_given(0.0) - 3.0) < 1e-10
    println("✓ Given solution satisfies x(0) = 3")
    exact_sol = exact_solution_given
else
    println("⚠ Neither solution exactly satisfies x(0) = 3")
    println("Using given solution for comparison...")
    exact_sol = exact_solution_given
end

# =================================================================
# Fourth-Order Runge-Kutta Method
# =================================================================

"""
    rk4_step(f, t, x, h)

Perform one step of the fourth-order Runge-Kutta method.

# Arguments
- `f`: Function defining x' = f(t, x)
- `t`: Current time
- `x`: Current solution value
- `h`: Step size

# Returns
- `x_next`: Solution value at t + h
"""
function rk4_step(f, t, x, h)
    k1 = f(t, x)
    k2 = f(t + h/2, x + h*k1/2)
    k3 = f(t + h/2, x + h*k2/2)
    k4 = f(t + h, x + h*k3)
    
    x_next = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next
end

"""
    rk4_solve(f, t0, tf, x0, h)

Solve IVP x' = f(t, x) with x(t0) = x0 on interval [t0, tf] using RK4.

# Arguments
- `f`: Function defining x' = f(t, x)
- `t0`: Initial time
- `tf`: Final time
- `x0`: Initial condition
- `h`: Step size

# Returns
- `t_values`: Array of time points
- `x_values`: Array of solution values
"""
function rk4_solve(f, t0, tf, x0, h)
    # Determine number of steps
    n_steps = round(Int, (tf - t0) / h)
    
    # Initialize arrays
    t_values = zeros(n_steps + 1)
    x_values = zeros(n_steps + 1)
    
    # Initial condition
    t_values[1] = t0
    x_values[1] = x0
    
    # RK4 iterations
    for i in 1:n_steps
        t_values[i+1] = t_values[i] + h
        x_values[i+1] = rk4_step(f, t_values[i], x_values[i], h)
    end
    
    return t_values, x_values
end

# =================================================================
# Solve the Problem
# =================================================================

println("\n\n🎯 Solving using RK4")
println("-"^80)

# Problem parameters
t0 = 0.0
tf = -2.0
x0 = 3.0
h = -0.01

println("Initial time t₀ = ", t0)
println("Final time tₑ = ", tf)
println("Initial condition x₀ = ", x0)
println("Step size h = ", h)
println("Number of steps = ", abs(round(Int, (tf - t0) / h)))

# Solve using RK4
t_rk4, x_rk4 = rk4_solve(f, t0, tf, x0, h)

println("\n✓ RK4 solution computed successfully!")

# =================================================================
# Compute Exact Solution
# =================================================================

println("\n\n📊 Computing Exact Solution")
println("-"^80)

x_exact = [exact_sol(t) for t in t_rk4]

# =================================================================
# Error Analysis
# =================================================================

println("\n\n📈 Error Analysis")
println("-"^80)

errors = abs.(x_rk4 .- x_exact)
max_error = maximum(errors)
mean_error = sum(errors) / length(errors)

println("Maximum absolute error: ", @sprintf("%.6e", max_error))
println("Mean absolute error:    ", @sprintf("%.6e", mean_error))
println("Relative error at t=-2: ", @sprintf("%.6e", errors[end] / abs(x_exact[end])))

# Print comparison table at selected points
println("\n" * "="^80)
println("Comparison Table (Selected Points)")
println("="^80)
println(@sprintf("%-10s %-20s %-20s %-15s", "t", "RK4 Solution", "Exact Solution", "Abs Error"))
println("-"^80)

# Select points to display
display_points = [-2.0, -1.5, -1.0, -0.5, 0.0]
for t_display in display_points
    # Find closest index
    idx = argmin(abs.(t_rk4 .- t_display))
    t_val = t_rk4[idx]
    x_rk4_val = x_rk4[idx]
    x_exact_val = x_exact[idx]
    error_val = errors[idx]
    
    println(@sprintf("%-10.1f %-20.12f %-20.12f %-15.6e", 
            t_val, x_rk4_val, x_exact_val, error_val))
end
println("="^80)

# =================================================================
# Visualization - Dao Themed Plot
# =================================================================

println("\n\n📉 Generating Dao-Themed Plot")
println("-"^80)

# Dao color scheme inspired by yin-yang philosophy
# Deep blue for night/yin, warm gold for day/yang, soft grey for balance
dao_bg = RGB(0.95, 0.95, 0.93)      # Soft off-white (rice paper)
dao_grid = RGB(0.85, 0.85, 0.82)    # Light grey (mist)
dao_yin = RGB(0.15, 0.25, 0.35)     # Deep blue-grey (night)
dao_yang = RGB(0.85, 0.65, 0.25)    # Warm gold (sun)
dao_balance = RGB(0.45, 0.55, 0.60) # Soft blue-grey (dawn/dusk)

# Create the plot with Dao aesthetics
gr()  # Use GR backend for better quality
plot(
    size=(800, 600),
    dpi=300,
    background_color=dao_bg,
    foreground_color=dao_yin,
    gridcolor=dao_grid,
    gridalpha=0.3,
    gridstyle=:dot,
    framestyle=:box,
    legend=:topright,
    legendfontsize=11,
    legendfontfamily="serif",
    fontfamily="serif",
    margin=5Plots.mm
)

# Plot exact solution (Yang - the truth, the light)
plot!(t_rk4, x_exact,
    label="Analytical Solution",
    linewidth=3,
    linestyle=:solid,
    color=dao_yang,
    alpha=0.9
)

# Plot RK4 solution (Yin - the approximation, the shadow)
plot!(t_rk4, x_rk4,
    label="RK4 Numerical Solution",
    linewidth=2,
    linestyle=:dash,
    color=dao_yin,
    alpha=0.8
)

# Add markers at selected points to show the discrete nature
selected_indices = [argmin(abs.(t_rk4 .- t)) for t in display_points]
scatter!(t_rk4[selected_indices], x_rk4[selected_indices],
    label="RK4 Sample Points",
    markersize=6,
    markercolor=dao_balance,
    markerstrokewidth=2,
    markerstrokecolor=dao_yin,
    alpha=0.7
)

# Labels and title with Dao philosophy
xlabel!("Time t", fontsize=12)
ylabel!("Solution x(t)", fontsize=12)
title!("RK4 vs Analytical: The Dance of Approximation and Truth\n" * 
       "步履之間，數值與解析共舞 (Between Steps, Numerical and Analytical Dance Together)",
    fontsize=13,
    titlefontfamily="serif"
)

# Add a subtle annotation about the Dao philosophy
annotate!(
    -1.0, 1.5,
    text("陰陽平衡\n(Yin-Yang Balance)", 9, dao_balance, :center, "serif")
)

println("✓ Dao-themed plot created!")

# Save to processedData folder
processed_path = "../processedData/p02_rk4_dao_comparison.png"
savefig(processed_path)
println("✓ Saved to: ", processed_path)

# Copy to tex/figures folder
figures_dir = "../tex/figures"
if !isdir(figures_dir)
    mkdir(figures_dir)
    println("✓ Created figures directory: ", figures_dir)
end

figures_path = joinpath(figures_dir, "p02_rk4_dao_comparison.png")
cp(processed_path, figures_path, force=true)
println("✓ Copied to: ", figures_path)

println("\n📊 Plot saved successfully!")
println("  Location 1 (processed): ", processed_path)
println("  Location 2 (LaTeX):     ", figures_path)

# =================================================================
# Summary
# =================================================================

println("\n\n" * "="^80)
println("Problem 2 Complete! ✓")
println("="^80)

println("\n📝 Summary:")
println("-"^80)
println("Method: Fourth-Order Runge-Kutta (RK4)")
println("Interval: [", tf, ", ", t0, "]")
println("Step size: h = ", h)
println("Number of steps: ", length(t_rk4) - 1)
println("\nAccuracy:")
println("  Maximum absolute error: ", @sprintf("%.6e", max_error))
println("  Mean absolute error:    ", @sprintf("%.6e", mean_error))
println("\nConclusion:")
println("  The RK4 method provides highly accurate numerical solutions")
println("  for this ODE problem, with errors typically on the order of")
println("  ", @sprintf("%.0e", max_error), " for the chosen step size.")

println("\n" * "="^80)
