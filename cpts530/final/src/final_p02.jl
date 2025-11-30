# CPTS530 Final Project - Problem 2: Fourth-Order Runge-Kutta Method
# Author: Aryan Ritwajeet Jha
# Date: December 2025

# Load environment and utilities
include("preamble.jl")
include("test_utils.jl")

println(INFO, "="^80, RESET)
println(INFO, "CPTS530 Final Project - Problem 2: Fourth-Order Runge-Kutta Method", RESET)
println(INFO, "="^80, RESET)

# =================================================================
# Problem Setup
# =================================================================

println(INFO, "\n📋 Problem Statement", RESET)
println(INFO, "-"^80, RESET)
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
println(INFO, "\n🔍 Verifying Exact Solutions at t=0", RESET)
println(INFO, "-"^80, RESET)
println("Initial condition: x(0) = 3")
println("Our derived solution at t=0: ", @sprintf("%.10f", exact_solution(0.0)))
println("Given solution at t=0:       ", @sprintf("%.10f", exact_solution_given(0.0)))

# Check which one satisfies the initial condition
if abs(exact_solution(0.0) - 3.0) < 1e-10
    println(SUCCESS, "✓ Our derived solution satisfies x(0) = 3", RESET)
    exact_sol = exact_solution
elseif abs(exact_solution_given(0.0) - 3.0) < 1e-10
    println(SUCCESS, "✓ Given solution satisfies x(0) = 3", RESET)
    exact_sol = exact_solution_given
else
    println(WARNING, "⚠ Neither solution exactly satisfies x(0) = 3", RESET)
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

println(INFO, "\n\n🎯 Solving using RK4", RESET)
println(INFO, "-"^80, RESET)

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

println(SUCCESS, "\n✓ RK4 solution computed successfully!", RESET)

# =================================================================
# Compute Exact Solution
# =================================================================

println(INFO, "\n\n📊 Computing Exact Solution", RESET)
println(INFO, "-"^80, RESET)

x_exact = [exact_sol(t) for t in t_rk4]

# =================================================================
# Error Analysis
# =================================================================

println(INFO, "\n\n📈 Error Analysis", RESET)
println(INFO, "-"^80, RESET)

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
# Visualization
# =================================================================

println(INFO, "\n\n📉 Generating Plots", RESET)
println(INFO, "-"^80, RESET)

# Plot 1: Solution comparison
p1 = plot(t_rk4, x_rk4, 
         label="RK4 Solution",
         linewidth=2,
         xlabel="t",
         ylabel="x(t)",
         title="Comparison of RK4 and Exact Solutions",
         legend=:topright)
plot!(p1, t_rk4, x_exact,
      label="Exact Solution",
      linewidth=2,
      linestyle=:dash)
scatter!(p1, [0.0], [3.0],
         label="Initial Condition",
         markersize=6,
         color=:red)

# Plot 2: Error over time
p2 = plot(t_rk4, errors,
         label="Absolute Error",
         linewidth=2,
         xlabel="t",
         ylabel="|x_RK4 - x_exact|",
         title="Absolute Error vs Time",
         legend=:topright,
         yaxis=:log)

# Plot 3: Relative error
rel_errors = errors ./ abs.(x_exact)
p3 = plot(t_rk4, rel_errors,
         label="Relative Error",
         linewidth=2,
         xlabel="t",
         ylabel="|x_RK4 - x_exact| / |x_exact|",
         title="Relative Error vs Time",
         legend=:topright,
         yaxis=:log)

# Combine plots
combined_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 900))

# Save plot
output_dir = "../tex/figures"
if !isdir(output_dir)
    mkpath(output_dir)
end

savefig(combined_plot, joinpath(output_dir, "p02_rk4_comparison.pdf"))
println(SUCCESS, "✓ Plot saved to: ", joinpath(output_dir, "p02_rk4_comparison.pdf"), RESET)

# =================================================================
# Summary
# =================================================================

println(INFO, "\n\n" * "="^80, RESET)
println(SUCCESS, "Problem 2 Complete! ✓", RESET)
println(INFO, "="^80, RESET)

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

println(INFO, "\n" * "="^80, RESET)
