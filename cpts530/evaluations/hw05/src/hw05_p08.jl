# CPTS530 HW05 - Problem 8: Polynomial Interpolation
# Author: Aryan Ritwajeet Jha
# Date: December 2025

# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using LinearAlgebra
using Printf
using Crayons

println("="^80)
println("CPTS530 HW05 - Problem 8: Polynomial Interpolation")
println("="^80)

# =================================================================
# Problem: Find polynomial of least degree interpolating data points
# Data: (3, 12), (7, 146), (12, 2)
# =================================================================

println("\n📊 Problem Statement")
println("-"^80)
println("Find the polynomial of least degree that interpolates:")
println("  x = [3, 7, 12]")
println("  y = [12, 146, 2]")

# Data points
x_data = [3.0, 7.0, 12.0]
y_data = [12.0, 146.0, 2.0]
n = length(x_data)

println("\n✓ Number of data points: $n")
println("✓ Degree of interpolating polynomial: $(n-1)")

# =================================================================
# Method 1: Lagrange Interpolation
# =================================================================

println("\n\n🔍 Method 1: Lagrange Interpolation")
println("-"^80)

"""
    lagrange_basis(x_data, i, x)

Compute the i-th Lagrange basis polynomial Lᵢ(x)
"""
function lagrange_basis(x_data, i, x)
    n = length(x_data)
    L = 1.0
    for j in 1:n
        if j != i
            L *= (x - x_data[j]) / (x_data[i] - x_data[j])
        end
    end
    return L
end

"""
    lagrange_interpolation(x_data, y_data, x)

Evaluate the Lagrange interpolating polynomial at point x
P(x) = Σ yᵢ * Lᵢ(x)
"""
function lagrange_interpolation(x_data, y_data, x)
    n = length(x_data)
    P = 0.0
    for i in 1:n
        P += y_data[i] * lagrange_basis(x_data, i, x)
    end
    return P
end

# Verify interpolation at data points
println("\nVerifying interpolation at data points:")
for i in 1:n
    P_xi = lagrange_interpolation(x_data, y_data, x_data[i])
    error = abs(P_xi - y_data[i])
    println(@sprintf("  P(%g) = %.6f  (expected: %g, error: %.2e)", 
            x_data[i], P_xi, y_data[i], error))
end

# =================================================================
# Method 2: Vandermonde Matrix (for finding coefficients)
# =================================================================

println("\n\n🔍 Method 2: Vandermonde Matrix System")
println("-"^80)

# Construct Vandermonde matrix
# For polynomial p(x) = a₀ + a₁x + a₂x²
# [1  x₁  x₁²] [a₀]   [y₁]
# [1  x₂  x₂²] [a₁] = [y₂]
# [1  x₃  x₃²] [a₂]   [y₃]

V = zeros(n, n)
for i in 1:n
    for j in 1:n
        V[i, j] = x_data[i]^(j-1)
    end
end

println("\nVandermonde matrix V:")
display(V)

# Solve for coefficients
coeffs = V \ y_data

println("\n\nPolynomial coefficients [a₀, a₁, a₂]:")
for (i, c) in enumerate(coeffs)
    println(@sprintf("  a%d = %.10f", i-1, c))
end

# Display the polynomial
println("\n📝 Interpolating Polynomial:")
println("-"^80)
println(@sprintf("P(x) = %.6f + %.6f·x + %.6f·x²", coeffs[1], coeffs[2], coeffs[3]))

# Simplify display if possible
if abs(coeffs[3]) < 1e-10
    if abs(coeffs[2]) < 1e-10
        println(@sprintf("\nSimplified: P(x) = %.6f (constant)", coeffs[1]))
    else
        println(@sprintf("\nSimplified: P(x) = %.6f + %.6f·x (linear)", coeffs[1], coeffs[2]))
    end
else
    println("\nThis is a quadratic polynomial (degree 2)")
end

# =================================================================
# Verification and Analysis
# =================================================================

println("\n\n✓ Verification")
println("-"^80)

"""
    evaluate_polynomial(coeffs, x)

Evaluate polynomial with given coefficients at x
"""
function evaluate_polynomial(coeffs, x)
    result = 0.0
    for (i, c) in enumerate(coeffs)
        result += c * x^(i-1)
    end
    return result
end

println("\nVerifying polynomial at data points:")
global max_error = 0.0
for i in 1:n
    P_xi = evaluate_polynomial(coeffs, x_data[i])
    error = abs(P_xi - y_data[i])
    global max_error = max(max_error, error)
    status = error < 1e-10 ? "✓" : "✗"
    println(@sprintf("  %s P(%g) = %.10f  (expected: %g, error: %.2e)", 
            status, x_data[i], P_xi, y_data[i], error))
end

println(@sprintf("\nMaximum interpolation error: %.2e", max_error))

if max_error < 1e-10
    println(Crayon(foreground=:green, bold=true), 
            "\n✓ SUCCESS: Polynomial correctly interpolates all data points!", 
            Crayon(reset=true))
else
    println(Crayon(foreground=:yellow, bold=true), 
            "\n⚠ WARNING: Numerical errors detected", 
            Crayon(reset=true))
end

# =================================================================
# Additional Analysis: Evaluate at intermediate points
# =================================================================

println("\n\n📈 Additional Evaluation")
println("-"^80)
println("\nEvaluating polynomial at some intermediate points:")

test_points = [5.0, 9.0, 10.0]
for x_test in test_points
    y_test = evaluate_polynomial(coeffs, x_test)
    println(@sprintf("  P(%.1f) = %.6f", x_test, y_test))
end

# =================================================================
# Save results
# =================================================================

println("\n\n💾 Saving Results")
println("-"^80)

output_file = joinpath(@__DIR__, "p08_output.txt")
open(output_file, "w") do io
    println(io, "CPTS530 HW05 - Problem 8: Polynomial Interpolation")
    println(io, "="^80)
    println(io, "\nData Points:")
    println(io, "  x = [3, 7, 12]")
    println(io, "  y = [12, 146, 2]")
    
    println(io, "\n\nInterpolating Polynomial (degree 2):")
    println(io, @sprintf("P(x) = %.10f + %.10f·x + %.10f·x²", 
            coeffs[1], coeffs[2], coeffs[3]))
    
    println(io, "\n\nCoefficients:")
    for (i, c) in enumerate(coeffs)
        println(io, @sprintf("  a%d = %.10f", i-1, c))
    end
    
    println(io, "\n\nVerification at data points:")
    for i in 1:n
        P_xi = evaluate_polynomial(coeffs, x_data[i])
        error = abs(P_xi - y_data[i])
        println(io, @sprintf("  P(%g) = %.10f  (expected: %g, error: %.2e)", 
                x_data[i], P_xi, y_data[i], error))
    end
    
    println(io, @sprintf("\nMaximum interpolation error: %.2e", max_error))
end

println("✓ Results saved to: $output_file")

println("\n" * "="^80)
println("Problem 8 Complete!")
println("="^80)
