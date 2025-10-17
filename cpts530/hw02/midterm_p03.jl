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
# Quick LU Factorization Tests
# =============================================================================

function test_LU_only(A; name="Test", show_matrices=true)
    """Quick test for LU factorization only (not solving systems)"""
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "Testing LU Factorization: $name", RESET)
    println(INFO, "Matrix size: $(size(A, 1)) × $(size(A, 2))", RESET)
    println(INFO, "=" ^ 80, RESET)
    
    println("\nOriginal matrix A:")
    display(A)
    
    try
        # Get LU factors
        L, U = get_LU_factors(A)
        
        if show_matrices
            println("\nL (Lower triangular):")
            display(L)
            println("\nU (Upper triangular):")
            display(U)
        end
        
        # Verify A = L*U
        println("\nVerifying A = L*U:")
        LU_product = L * U
        error_matrix = A - LU_product
        max_error = maximum(abs.(error_matrix))
        
        println("L * U:")
        display(LU_product)
        println("\nError matrix (A - L*U):")
        display(error_matrix)
        println("\nMax absolute error: ", @sprintf("%.3e", max_error))
        
        if max_error < 1e-10
            println(SUCCESS, "\n✓ LU factorization CORRECT! (max error < 1e-10)", RESET)
            return true
        elseif max_error < 1e-6
            println(WARNING, "\n⚠ LU factorization acceptable (max error < 1e-6)", RESET)
            return true
        else
            println(FAILURE, "\n✗ LU factorization INCORRECT! (max error = ", @sprintf("%.3e", max_error), ")", RESET)
            return false
        end
        
    catch e
        println(FAILURE, "\n✗ Error during LU factorization: ", e, RESET)
        println(FAILURE, "Stacktrace:", RESET)
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
end

# =============================================================================
# Main execution
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "MIDTERM PROBLEM 3: LU FACTORIZATION TESTS", RESET)
    println(INFO, "=" ^ 80, RESET)
    
    # Test Case 1: Simple 2×2
    println(INFO, "\n\nTEST 1: Simple 2×2 Matrix", RESET)
    A1 = [4.0  3.0;
          6.0  3.0]
    test_LU_only(A1, name="2×2 system")
    
    println("\n", INFO, "Expected L:", RESET)
    println("[1.0   0.0]")
    println("[1.5   1.0]")
    println(INFO, "Expected U:", RESET)
    println("[4.0   3.0]")
    println("[0.0  -1.5]")
    
    # Test Case 2: Problem from midterm (3×3)
    println(INFO, "\n\nTEST 2: Midterm Problem 3×3 Matrix", RESET)
    A2 = [2.0   1.0  1.0;
          4.0  -6.0  0.0;
         -2.0   7.0  2.0]
    test_LU_only(A2, name="Midterm 3×3 system")
    
    println("\n", INFO, "Expected L:", RESET)
    println("[1.0   0.0  0.0]")
    println("[2.0   1.0  0.0]")
    println("[-1.0 -1.0  1.0]")
    println(INFO, "Expected U:", RESET)
    println("[2.0  1.0   1.0]")
    println("[0.0 -8.0  -2.0]")
    println("[0.0  0.0   1.0]")
    
    # Test Case 3: Identity matrix (should give L=I, U=I)
    println(INFO, "\n\nTEST 3: Identity Matrix", RESET)
    A3 = Matrix{Float64}(I, 3, 3)
    test_LU_only(A3, name="Identity 3×3")
    
    # Test Case 4: Diagonal matrix
    println(INFO, "\n\nTEST 4: Diagonal Matrix", RESET)
    A4 = [2.0  0.0  0.0;
          0.0  3.0  0.0;
          0.0  0.0  4.0]
    test_LU_only(A4, name="Diagonal 3×3")
    
    println(INFO, "\n\n" * "=" ^ 80, RESET)
    println(INFO, "LU FACTORIZATION TESTING COMPLETE", RESET)
    println(INFO, "=" ^ 80, RESET)
    println(INFO, "\nNext steps:", RESET)
    println(INFO, "  - If all tests pass, implement solve_Ly_is_equal_to_b", RESET)
    println(INFO, "  - Then implement solve_Ux_is_equal_to_y", RESET)
    println(INFO, "  - Finally use test_problem_1(), test_problem_2(), etc.", RESET)
    println(INFO, "=" ^ 80, RESET)
end
