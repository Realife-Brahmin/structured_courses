# CPTS530 HW05 - Problem 7: Gauss-Seidel Method
# Author: Aryan Ritwajeet Jha
# Date: December 2025

# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using LinearAlgebra
using Printf

println("="^80)
println("CPTS530 HW05 - Problem 7: Gauss-Seidel Method")
println("="^80)

# =================================================================
# Problem: Solve system using Gauss-Seidel and analyze Gaussian elimination
# System:
#   3x + y + z = 5
#   3x + y - 5z = -1
#   x + 3y - z = 3
# =================================================================

println("\n📊 Problem Statement")
println("-"^80)
println("System of equations:")
println("  3x + y + z = 5")
println("  3x + y - 5z = -1")
println("  x + 3y - z = 3")

# Coefficient matrix A and right-hand side b
A = [3.0  1.0   1.0;
     3.0  1.0  -5.0;
     1.0  3.0  -1.0]

b = [5.0; -1.0; 3.0]

println("\nCoefficient matrix A:")
display(A)
println("\n\nRight-hand side b:")
display(b)

# =================================================================
# Part 1: True solution using direct method
# =================================================================

println("\n\n🎯 Part 1: True Solution (Direct Method)")
println("-"^80)

x_true = A \ b
println("True solution:")
println(@sprintf("  x = %.10f", x_true[1]))
println(@sprintf("  y = %.10f", x_true[2]))
println(@sprintf("  z = %.10f", x_true[3]))

# Verify
residual = A * x_true - b
println("\nResidual ||Ax - b||: ", norm(residual))

# =================================================================
# Part 2: Gauss-Seidel Method
# =================================================================

println("\n\n🔄 Part 2: Gauss-Seidel Method")
println("-"^80)

"""
    gauss_seidel(A, b, x0; max_iter=100, tol=1e-10)

Solve Ax = b using Gauss-Seidel iteration.
Returns (x, converged, iterations, history)
"""
function gauss_seidel(A, b, x0; max_iter=100, tol=1e-10)
    n = length(b)
    x = copy(x0)
    history = [copy(x)]
    
    for iter in 1:max_iter
        x_old = copy(x)
        
        # Gauss-Seidel iteration
        for i in 1:n
            sum_val = b[i]
            for j in 1:n
                if j != i
                    sum_val -= A[i, j] * x[j]
                end
            end
            x[i] = sum_val / A[i, i]
        end
        
        push!(history, copy(x))
        
        # Check convergence
        error = norm(x - x_old, Inf)
        if error < tol
            return (x, true, iter, history)
        end
    end
    
    return (x, false, max_iter, history)
end

# Check diagonal dominance
println("\nChecking diagonal dominance:")
global diag_dominant = true
for i in 1:3
    row_sum = sum(abs.(A[i, :])) - abs(A[i, i])
    is_dominant = abs(A[i, i]) > row_sum
    println(@sprintf("  Row %d: |a_%d%d| = %.1f, sum of |others| = %.1f  %s", 
            i, i, i, abs(A[i, i]), row_sum, is_dominant ? "✓" : "✗"))
    global diag_dominant = diag_dominant && is_dominant
end

if !diag_dominant
    println("\n⚠ WARNING: Matrix is NOT strictly diagonally dominant!")
    println("  Gauss-Seidel may not converge.")
end

# Try multiple starting points
starting_points = [
    ([0.0, 0.0, 0.0], "zeros"),
    ([1.0, 1.0, 1.0], "ones"),
    ([5.0, -1.0, 3.0], "b vector"),
    ([10.0, 10.0, 10.0], "far from solution")
]

println("\n\nTrying Gauss-Seidel with different starting points:")
println("-"^80)

for (x0, name) in starting_points
    println("\nStarting point: $name = $x0")
    x_gs, converged, iters, history = gauss_seidel(A, b, x0, max_iter=1000, tol=1e-10)
    
    if converged
        error_norm = norm(x_gs - x_true)
        println(@sprintf("  ✓ Converged in %d iterations", iters))
        println(@sprintf("  Solution: x = %.6f, y = %.6f, z = %.6f", x_gs[1], x_gs[2], x_gs[3]))
        println(@sprintf("  Error from true solution: %.2e", error_norm))
    else
        println("  ✗ Did NOT converge after 1000 iterations")
        println(@sprintf("  Last iterate: x = %.6f, y = %.6f, z = %.6f", x_gs[1], x_gs[2], x_gs[3]))
        error_norm = norm(x_gs - x_true)
        println(@sprintf("  Error from true solution: %.2e", error_norm))
    end
end

# =================================================================
# Part 3: Gaussian Elimination without Pivoting
# =================================================================

println("\n\n📐 Part 3: Gaussian Elimination WITHOUT Pivoting")
println("-"^80)

"""
    gaussian_elimination_no_pivot(A, b)

Solve Ax = b using Gaussian elimination without pivoting.
Returns (x, success, A_final, b_final)
"""
function gaussian_elimination_no_pivot(A, b)
    n = length(b)
    A_work = copy(A)
    b_work = copy(b)
    
    # Forward elimination
    for k in 1:n-1
        println(@sprintf("\nStep %d: Eliminate column %d", k, k))
        
        if abs(A_work[k, k]) < 1e-14
            println("  ⚠ WARNING: Pivot element is near zero!")
            println(@sprintf("    a_%d%d = %.2e", k, k, A_work[k, k]))
            return (zeros(n), false, A_work, b_work)
        end
        
        println(@sprintf("  Pivot: a_%d%d = %.6f", k, k, A_work[k, k]))
        
        for i in k+1:n
            if abs(A_work[i, k]) > 1e-14
                multiplier = A_work[i, k] / A_work[k, k]
                println(@sprintf("  Row %d: subtract %.6f × Row %d", i, multiplier, k))
                
                A_work[i, k:n] -= multiplier * A_work[k, k:n]
                b_work[i] -= multiplier * b_work[k]
            end
        end
        
        println("\n  Matrix after step $k:")
        display(A_work)
        println("\n  RHS after step $k:")
        display(b_work)
    end
    
    # Back substitution
    x = zeros(n)
    for i in n:-1:1
        x[i] = (b_work[i] - dot(A_work[i, i+1:n], x[i+1:n])) / A_work[i, i]
    end
    
    return (x, true, A_work, b_work)
end

println("\nPerforming Gaussian elimination without pivoting:")
x_gauss, success, A_final, b_final = gaussian_elimination_no_pivot(A, b)

if success
    println("\n\n✓ Gaussian elimination completed successfully")
    println("\nFinal upper triangular matrix:")
    display(A_final)
    println("\n\nSolution:")
    println(@sprintf("  x = %.10f", x_gauss[1]))
    println(@sprintf("  y = %.10f", x_gauss[2]))
    println(@sprintf("  z = %.10f", x_gauss[3]))
    
    error_norm = norm(x_gauss - x_true)
    println(@sprintf("\nError from true solution: %.2e", error_norm))
    
    residual_gauss = A * x_gauss - b
    println(@sprintf("Residual ||Ax - b||: %.2e", norm(residual_gauss)))
else
    println("\n✗ Gaussian elimination failed (zero pivot encountered)")
end

# =================================================================
# Part 4: Analysis and Comparison
# =================================================================

println("\n\n📊 Part 4: Analysis and Comparison")
println("-"^80)

println("\n1. Matrix Properties:")
println(@sprintf("   Condition number: %.2e", cond(A)))
println(@sprintf("   Determinant: %.6f", det(A)))

println("\n2. Why Gauss-Seidel may not converge:")
println("   - Matrix is NOT strictly diagonally dominant")
println("   - Row 2: |1.0| < |3.0| + |-5.0| = 8.0")
println("   - Gauss-Seidel convergence is not guaranteed")

println("\n3. Gaussian Elimination without pivoting:")
if success
    println("   - Succeeds because no zero pivots are encountered")
    println("   - However, small pivots can lead to numerical instability")
else
    println("   - Fails due to zero or near-zero pivot")
end

# =================================================================
# Save results
# =================================================================

println("\n\n💾 Saving Results")
println("-"^80)

output_file = joinpath(@__DIR__, "p07_output.txt")
open(output_file, "w") do io
    println(io, "CPTS530 HW05 - Problem 7: Gauss-Seidel Method")
    println(io, "="^80)
    
    println(io, "\nSystem:")
    println(io, "  3x + y + z = 5")
    println(io, "  3x + y - 5z = -1")
    println(io, "  x + 3y - z = 3")
    
    println(io, "\n\nTrue Solution (Direct Method):")
    println(io, @sprintf("  x = %.10f", x_true[1]))
    println(io, @sprintf("  y = %.10f", x_true[2]))
    println(io, @sprintf("  z = %.10f", x_true[3]))
    
    println(io, "\n\nGauss-Seidel Method:")
    println(io, "  Matrix is NOT strictly diagonally dominant")
    println(io, "  Convergence is NOT guaranteed")
    
    println(io, "\n\nGaussian Elimination without Pivoting:")
    if success
        println(io, "  Status: SUCCESS")
        println(io, @sprintf("  x = %.10f", x_gauss[1]))
        println(io, @sprintf("  y = %.10f", x_gauss[2]))
        println(io, @sprintf("  z = %.10f", x_gauss[3]))
        println(io, @sprintf("  Error: %.2e", norm(x_gauss - x_true)))
    else
        println(io, "  Status: FAILED (zero pivot)")
    end
    
    println(io, "\n\nMatrix Properties:")
    println(io, @sprintf("  Condition number: %.2e", cond(A)))
    println(io, @sprintf("  Determinant: %.6f", det(A)))
end

println("✓ Results saved to: $output_file")

println("\n" * "="^80)
println("Problem 7 Complete!")
println("="^80)
