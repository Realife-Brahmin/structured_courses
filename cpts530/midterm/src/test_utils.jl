# Testing and verification utilities
# This file contains functions for testing LU factorization and system solving

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

"""
    test_LU_only(A; name="Test", show_matrices=true)

Quick test for LU factorization only (not solving systems).

# Arguments
- `A::Matrix`: Matrix to factorize
- `name::String`: Name for the test case
- `show_matrices::Bool`: Whether to display L and U matrices (default: true)

# Returns
- `Bool`: true if factorization is correct
"""
function test_LU_only(A; name="Test", show_matrices=true)
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
