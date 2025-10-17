# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using Printf
using Crayons

# Define color schemes
const SUCCESS = Crayon(foreground=:green, bold=true)
const FAILURE = Crayon(foreground=:red, bold=true)
const INFO = Crayon(foreground=:cyan, bold=true)
const WARNING = Crayon(foreground=:yellow, bold=true)
const RESET = Crayon(reset=true)

"""
    get_LU_factors(A)

Compute LU factorization of matrix A without pivoting.
Returns L (lower triangular) and U (upper triangular) such that A = LU.

# Arguments
- `A::Matrix`: Input square matrix

# Returns
- `L::Matrix`: Lower triangular matrix with ones on diagonal
- `U::Matrix`: Upper triangular matrix
"""
function get_LU_factors(A)
    # TODO: Implement LU factorization
    n = size(A, 1)
    L = zeros(n, n)
    U = zeros(n, n)
    
    for r = 1:n
        L[r, r] = 1.0
        for c = 1:n
            if r > c # L values before diagonal need to be computed
                L[r, c] = (A[r, c] - sum([L[r, k]*U[k, c] for k ∈ 1:c-1]))/U[c, c]
            elseif c >= r # U values after diagonal need to be computed
                U[r, c] = A[r, c] - sum([L[r, k]*U[k, c] for k ∈ 1:r-1])
            end
        end
    end
    
    return L, U
end

A = [2.0 1.0 1.0;
    4.0 -6.0 0.0;
    -2.0 7.0 2.0]
L, U  = get_LU_factors(A);
display(L);
display(U);

"""
    solve_Ly_is_equal_to_b(L, b)

Solve Ly = b for y using forward substitution.
Assumes L is lower triangular with ones on the diagonal.

# Arguments
- `L::Matrix`: Lower triangular matrix
- `b::Vector`: Right-hand side vector

# Returns
- `y::Vector`: Solution vector
"""
function solve_Ly_is_equal_to_b(L, b)
    # TODO: Implement forward substitution
    n = length(b)
    y = zeros(n)
    
    # Your implementation here
    # Forward substitution: y[i] = (b[i] - sum(L[i,j]*y[j] for j=1:i-1)) / L[i,i]
    
    return y
end

"""
    solve_Ux_is_equal_to_y(U, y)

Solve Ux = y for x using backward substitution.
Assumes U is upper triangular.

# Arguments
- `U::Matrix`: Upper triangular matrix
- `y::Vector`: Right-hand side vector

# Returns
- `x::Vector`: Solution vector
"""
function solve_Ux_is_equal_to_y(U, y)
    # TODO: Implement backward substitution
    n = length(y)
    x = zeros(n)
    
    # Your implementation here
    # Backward substitution: x[i] = (y[i] - sum(U[i,j]*x[j] for j=i+1:n)) / U[i,i]
    
    return x
end

"""
    solve_linear_system_LU(A, b)

Solve Ax = b using LU factorization.
This is the main driver function that uses get_LU_factors, 
solve_Ly_is_equal_to_b, and solve_Ux_is_equal_to_y.

# Arguments
- `A::Matrix`: Coefficient matrix
- `b::Vector`: Right-hand side vector

# Returns
- `x::Vector`: Solution vector
- `L::Matrix`: Lower triangular factor
- `U::Matrix`: Upper triangular factor
"""
function solve_linear_system_LU(A, b)
    # Step 1: Get LU factors
    L, U = get_LU_factors(A)
    
    # Step 2: Solve Ly = b
    y = solve_Ly_is_equal_to_b(L, b)
    
    # Step 3: Solve Ux = y
    x = solve_Ux_is_equal_to_y(U, y)
    
    return x, L, U
end

"""
    verify_LU_factorization(A, L, U; tol=1e-10)

Verify that A = LU within tolerance.

# Arguments
- `A::Matrix`: Original matrix
- `L::Matrix`: Lower triangular factor
- `U::Matrix`: Upper triangular factor
- `tol::Float64`: Tolerance for comparison (default: 1e-10)

# Returns
- `Bool`: true if A ≈ LU within tolerance
"""
function verify_LU_factorization(A, L, U; tol=1e-10)
    LU_product = L * U
    max_error = maximum(abs.(A - LU_product))
    
    if max_error < tol
        println(SUCCESS, "✓ LU factorization verified: max error = ", @sprintf("%.3e", max_error), RESET)
        return true
    else
        println(FAILURE, "✗ LU factorization failed: max error = ", @sprintf("%.3e", max_error), RESET)
        return false
    end
end

"""
    verify_solution(A, x, b; tol=1e-10)

Verify that Ax ≈ b within tolerance.

# Arguments
- `A::Matrix`: Coefficient matrix
- `x::Vector`: Solution vector
- `b::Vector`: Right-hand side vector
- `tol::Float64`: Tolerance for comparison (default: 1e-10)

# Returns
- `Bool`: true if Ax ≈ b within tolerance
"""
function verify_solution(A, x, b; tol=1e-10)
    residual = A * x - b
    max_residual = maximum(abs.(residual))
    
    if max_residual < tol
        println(SUCCESS, "✓ Solution verified: max residual = ", @sprintf("%.3e", max_residual), RESET)
        return true
    else
        println(FAILURE, "✗ Solution verification failed: max residual = ", @sprintf("%.3e", max_residual), RESET)
        return false
    end
end

"""
    test_LU_solver(A, b; name="Test")

Test the LU solver on a given system and compare with built-in solver.

# Arguments
- `A::Matrix`: Coefficient matrix
- `b::Vector`: Right-hand side vector
- `name::String`: Name for the test case
"""
function test_LU_solver(A, b; name="Test")
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "Testing: $name", RESET)
    println(INFO, "Matrix size: $(size(A, 1)) × $(size(A, 2))", RESET)
    println(INFO, "=" ^ 80, RESET)
    
    # Solve using custom LU
    println("\nSolving Ax = b using custom LU factorization...")
    try
        x_custom, L, U = solve_linear_system_LU(A, b)
        
        println("\nVerifying LU factorization:")
        verify_LU_factorization(A, L, U)
        
        println("\nVerifying solution:")
        verify_solution(A, x_custom, b)
        
        # Compare with Julia's built-in solver
        println("\nComparing with Julia's built-in solver:")
        x_builtin = A \ b
        diff = maximum(abs.(x_custom - x_builtin))
        println("Max difference from built-in: ", @sprintf("%.3e", diff))
        
        if diff < 1e-10
            println(SUCCESS, "✓ Solutions match!", RESET)
        else
            println(WARNING, "⚠ Solutions differ by ", @sprintf("%.3e", diff), RESET)
        end
        
        # Print solutions side by side
        println("\nSolution comparison:")
        println(@sprintf("%-4s %-15s %-15s %-15s", "i", "x_custom", "x_builtin", "difference"))
        println("-" ^ 60)
        for i in 1:length(x_custom)
            println(@sprintf("%-4d %-15.6e %-15.6e %-15.6e", 
                            i, x_custom[i], x_builtin[i], abs(x_custom[i] - x_builtin[i])))
        end
        
    catch e
        println(FAILURE, "✗ Error during solving: ", e, RESET)
    end
end

# =============================================================================
# Test Problems
# =============================================================================

"""
Define test problems here
"""

# Simple 3x3 system
function test_problem_1()
    A = [2.0  1.0  1.0;
         4.0 -6.0  0.0;
        -2.0  7.0  2.0]
    b = [5.0, -2.0, 9.0]
    
    test_LU_solver(A, b, name="Problem 1: Simple 3×3 system")
end

# Another test problem
function test_problem_2()
    A = [4.0  3.0;
         6.0  3.0]
    b = [10.0, 12.0]
    
    test_LU_solver(A, b, name="Problem 2: Simple 2×2 system")
end

# Test with identity matrix (trivial case)
function test_problem_identity()
    n = 4
    A = Matrix{Float64}(I, n, n)
    b = ones(n)
    
    test_LU_solver(A, b, name="Test: Identity matrix")
end

# Test with diagonal matrix
function test_problem_diagonal()
    A = [2.0  0.0  0.0;
         0.0  3.0  0.0;
         0.0  0.0  4.0]
    b = [4.0, 9.0, 16.0]
    
    test_LU_solver(A, b, name="Test: Diagonal matrix")
end

# =============================================================================
# Main execution
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "MIDTERM PROBLEM 3: LU FACTORIZATION AND LINEAR SYSTEMS", RESET)
    println(INFO, "=" ^ 80, RESET)
    
    # Uncomment tests as you implement the functions
    # test_problem_1()
    # test_problem_2()
    # test_problem_identity()
    # test_problem_diagonal()
    
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "Implementation TODO:", RESET)
    println(INFO, "  1. Implement get_LU_factors(A)", RESET)
    println(INFO, "  2. Implement solve_Ly_is_equal_to_b(L, b)", RESET)
    println(INFO, "  3. Implement solve_Ux_is_equal_to_y(U, y)", RESET)
    println(INFO, "  4. Uncomment test problems to verify", RESET)
    println(INFO, "  5. Add more test problems from midterm", RESET)
    println(INFO, "=" ^ 80, RESET)
end
