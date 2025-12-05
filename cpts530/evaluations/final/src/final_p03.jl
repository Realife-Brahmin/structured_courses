# CPTS530 Final Project - Problem 3: Adams-Bashforth-Moulton Method
# Author: Aryan Ritwajeet Jha
# Date: December 2025

# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using LinearAlgebra
using Printf
using Plots

println("="^80)
println("CPTS530 Final Project - Problem 3: Adams-Bashforth-Moulton Method")
println("="^80)

# =================================================================
# Problem Setup
# =================================================================

println("\n📋 Problem Statement")
println("-"^80)
println("""
Solve the initial-value problem:
    y' = -2xy²
    y(0) = 1

on the interval [0, 1] using step size h = 0.25

Use 4th-order Adams-Bashforth-Moulton method with RK4 for bootstrap.
Compare with exact analytical solution: y = 1/(1 + x²)
""")

# =================================================================
# Define the ODE: y' = f(x, y)
# =================================================================

"""
    f(x, y)

Right-hand side of the ODE: y' = f(x, y)
f(x, y) = -2xy²
"""
function f(x, y)
    return -2 * x * y^2
end

# =================================================================
# Exact Analytical Solution
# =================================================================

"""
    exact_solution(x)

Exact analytical solution: y(x) = 1/(1 + x²)
"""
function exact_solution(x)
    return 1 / (1 + x^2)
end

# =================================================================
# Fourth-Order Runge-Kutta Method (for bootstrap)
# =================================================================

"""
    rk4_step(f, x, y, h)

Perform one step of the fourth-order Runge-Kutta method.

# Arguments
- `f`: Function defining y' = f(x, y)
- `x`: Current x value
- `y`: Current solution value
- `h`: Step size

# Returns
- `y_next`: Solution value at x + h
"""
function rk4_step(f, x, y, h)
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    
    y_next = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return y_next
end

# =================================================================
# Adams-Bashforth 4-step (Predictor)
# =================================================================

"""
    adams_bashforth_4(x_n, y_n, f_vals, h)

Fourth-order Adams-Bashforth predictor formula.

y*_{n+1} = y_n + (h/24)[55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3}]

# Arguments
- `x_n`: Current x value
- `y_n`: Current y value
- `f_vals`: Array [f_n, f_{n-1}, f_{n-2}, f_{n-3}]
- `h`: Step size

# Returns
- `y_pred`: Predicted value at x_{n+1}
"""
function adams_bashforth_4(x_n, y_n, f_vals, h)
    f_n, f_n1, f_n2, f_n3 = f_vals
    y_pred = y_n + (h/24) * (55*f_n - 59*f_n1 + 37*f_n2 - 9*f_n3)
    return y_pred
end

# =================================================================
# Adams-Moulton 4-step (Corrector)
# =================================================================

"""
    adams_moulton_4(x_n, y_n, y_pred, f_vals, h)

Fourth-order Adams-Moulton corrector formula.

y_{n+1} = y_n + (h/24)[9f_{n+1} + 19f_n - 5f_{n-1} + f_{n-2}]

# Arguments
- `x_n`: Current x value
- `y_n`: Current y value
- `y_pred`: Predicted value at x_{n+1}
- `f_vals`: Array [f_n, f_{n-1}, f_{n-2}]
- `h`: Step size

# Returns
- `y_corr`: Corrected value at x_{n+1}
"""
function adams_moulton_4(x_n, y_n, y_pred, f_vals, h)
    f_n, f_n1, f_n2 = f_vals
    f_n_plus_1 = f(x_n + h, y_pred)  # Evaluate f at predicted point
    y_corr = y_n + (h/24) * (9*f_n_plus_1 + 19*f_n - 5*f_n1 + f_n2)
    return y_corr
end

# =================================================================
# Complete ABM4 Solver
# =================================================================

"""
    abm4_solve(f, x0, xf, y0, h)

Solve IVP y' = f(x, y) with y(x0) = y0 on interval [x0, xf] using ABM4.

Uses RK4 for the first 3 steps, then switches to Adams-Bashforth-Moulton.

# Returns
- `x_values`: Array of x points
- `y_values`: Array of y values
"""
function abm4_solve(f, x0, xf, y0, h)
    # Determine number of steps
    n_steps = round(Int, (xf - x0) / h)
    
    # Initialize arrays
    x_values = zeros(n_steps + 1)
    y_values = zeros(n_steps + 1)
    f_values = zeros(n_steps + 1)  # Store f evaluations
    
    # Initial condition
    x_values[1] = x0
    y_values[1] = y0
    f_values[1] = f(x0, y0)
    
    println("\n🚀 Bootstrap Phase: Using RK4 for first 3 steps")
    println("-"^80)
    
    # Bootstrap with RK4 for first 3 steps
    for i in 1:min(3, n_steps)
        x_values[i+1] = x_values[i] + h
        y_values[i+1] = rk4_step(f, x_values[i], y_values[i], h)
        f_values[i+1] = f(x_values[i+1], y_values[i+1])
        
        println(@sprintf("Step %d (RK4): x = %.4f, y = %.10f", 
                        i, x_values[i+1], y_values[i+1]))
    end
    
    if n_steps <= 3
        return x_values, y_values
    end
    
    println("\n🎯 Main Phase: Using Adams-Bashforth-Moulton")
    println("-"^80)
    
    # Continue with ABM4 for remaining steps
    for i in 4:n_steps
        x_values[i+1] = x_values[i] + h
        
        # Predictor: Adams-Bashforth 4-step
        f_vals_ab = [f_values[i], f_values[i-1], f_values[i-2], f_values[i-3]]
        y_pred = adams_bashforth_4(x_values[i], y_values[i], f_vals_ab, h)
        
        # Corrector: Adams-Moulton 4-step
        f_vals_am = [f_values[i], f_values[i-1], f_values[i-2]]
        y_corr = adams_moulton_4(x_values[i], y_values[i], y_pred, f_vals_am, h)
        
        y_values[i+1] = y_corr
        f_values[i+1] = f(x_values[i+1], y_values[i+1])
        
        println(@sprintf("Step %d (ABM): x = %.4f, y_pred = %.10f, y_corr = %.10f", 
                        i, x_values[i+1], y_pred, y_corr))
    end
    
    return x_values, y_values
end

# =================================================================
# Solve the Problem
# =================================================================

println("\n\n🎯 Solving using ABM4")
println("-"^80)

# Problem parameters
x0 = 0.0
xf = 1.0
y0 = 1.0
h = 0.25

println("Initial value: y(", x0, ") = ", y0)
println("Final value x = ", xf)
println("Step size h = ", h)
println("Number of steps = ", round(Int, (xf - x0) / h))

# Solve using ABM4
x_abm, y_abm = abm4_solve(f, x0, xf, y0, h)

println("\n✓ ABM4 solution computed successfully!")

# =================================================================
# Compute Exact Solution
# =================================================================

println("\n\n📊 Computing Exact Solution")
println("-"^80)

y_exact = [exact_solution(x) for x in x_abm]

# =================================================================
# Comparison and Error Analysis
# =================================================================

println("\n\n📈 Comparison with Exact Solution")
println("-"^80)

errors = abs.(y_abm .- y_exact)
max_error = maximum(errors)
mean_error = sum(errors) / length(errors)

println("Maximum absolute error: ", @sprintf("%.6e", max_error))
println("Mean absolute error:    ", @sprintf("%.6e", mean_error))

# Print comparison table
println("\n" * "="^80)
println("Solution Comparison Table")
println("="^80)
println(@sprintf("%-10s %-20s %-20s %-15s %-15s", 
        "x", "ABM4 Solution", "Exact Solution", "Abs Error", "Rel Error (%)"))
println("-"^80)

for i in 1:length(x_abm)
    x_val = x_abm[i]
    y_abm_val = y_abm[i]
    y_exact_val = y_exact[i]
    error_abs = errors[i]
    error_rel = (error_abs / abs(y_exact_val)) * 100
    
    println(@sprintf("%-10.4f %-20.10f %-20.10f %-15.6e %-15.6e", 
            x_val, y_abm_val, y_exact_val, error_abs, error_rel))
end
println("="^80)

# =================================================================
# Generate Comparison Plot
# =================================================================

println("\n\n📉 Generating Comparison Plot")
println("-"^80)

# Create fine grid for exact solution
x_fine = range(x0, xf, length=100)
y_fine = exact_solution.(x_fine)

# Define color scheme (same as p02)
color_analytical = RGB(1.0, 0.4, 0.6)  # Pink/salmon for analytical
color_numerical = RGB(0.2, 0.5, 0.7)   # Blue for numerical
color_markers = RGB(0.3, 0.4, 0.6)     # Blue-grey for markers

# Create the plot
gr()
plot(
    size=(800, 600),
    dpi=300,
    legend=:topright,
    legendfontsize=11,
    margin=5Plots.mm
)

# Plot exact solution
plot!(x_fine, y_fine,
    label="Analytical Solution",
    linewidth=3,
    linestyle=:solid,
    color=color_analytical
)

# Plot ABM4 solution
plot!(x_abm, y_abm,
    label="ABM4 Numerical Solution",
    linewidth=2,
    linestyle=:dash,
    color=color_numerical
)

# Add markers at all ABM4 points
scatter!(x_abm, y_abm,
    label="ABM4 Sample Points",
    markersize=6,
    markercolor=color_markers
)

# Labels and title
xlabel!("x", fontsize=12)
ylabel!("y(x)", fontsize=12)
title!("ABM4 vs Analytical Solution",
    fontsize=13
)

# Save the plot
output_dir = joinpath(@__DIR__, "..", "processedData")
mkpath(output_dir)
output_file = joinpath(output_dir, "p03-solution-comparison.png")
savefig(output_file)
println("✓ Plot saved to: ", output_file)
println("✓ Plot saved to: ", output_file)

# Copy to figures directory
figures_dir = joinpath(@__DIR__, "..", "tex", "figures")
mkpath(figures_dir)
cp(output_file, joinpath(figures_dir, "p03-solution-comparison.png"), force=true)
println("✓ Plot copied to: ", joinpath(figures_dir, "p03-solution-comparison.png"))
# =================================================================
# Summary
# =================================================================

println("\n\n" * "="^80)
println("Problem 3 Complete! ✓")
println("="^80)

println("\n📝 Summary:")
println("-"^80)
println("Method: 4th-order Adams-Bashforth-Moulton")
println("Bootstrap: RK4 for first 3 steps")
println("Interval: [", x0, ", ", xf, "]")
println("Step size: h = ", h)
println("Number of steps: ", length(x_abm) - 1)
println("\nResults at requested points (5 significant digits):")
for x_val in [0.25, 0.5, 0.75, 1.0]
    idx = argmin(abs.(x_abm .- x_val))
    println(@sprintf("  y(%.2f) = %.5f", x_val, y_abm[idx]))
end
println("\nAccuracy:")
println("  Maximum absolute error: ", @sprintf("%.6e", max_error))
println("  Mean absolute error:    ", @sprintf("%.6e", mean_error))
println("\nConclusion:")
println("  The ABM4 method provides excellent accuracy for this ODE,")
println("  demonstrating the power of predictor-corrector methods.")

println("\n" * "="^80)
