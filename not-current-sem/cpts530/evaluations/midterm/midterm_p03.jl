# =============================================================================
# MIDTERM PROBLEM 3: LU Factorization Implementation
# =============================================================================
# This is the main file containing core LU factorization functions.
# Preamble and testing utilities are in separate files for modularity.

# Load environment setup and imports
include("src/preamble.jl")

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

"""
    solve_Ly_equals_b_for_y(L, b)

Solve Ly = b for y using forward substitution.
Assumes L is lower triangular with ones on the diagonal.

# Arguments
- `L::Matrix`: Lower triangular matrix
- `b::Vector`: Right-hand side vector

# Returns
- `y::Vector`: Solution vector
"""
function solve_Ly_equals_b_for_y(L, b)
    n = length(b)
    y = zeros(n)
    
    # Forward substitution: solve from top to bottom
    for i = 1:n
        sum_val = 0.0
        for j = 1:i-1
            sum_val += L[i,j] * y[j]
        end
        y[i] = (b[i] - sum_val) / L[i,i]
    end
    
    return y
end

"""
    solve_Ux_equals_y_for_x(U, y)

Solve Ux = y for x using backward substitution.
Assumes U is upper triangular.

# Arguments
- `U::Matrix`: Upper triangular matrix
- `y::Vector`: Right-hand side vector

# Returns
- `x::Vector`: Solution vector
"""
function solve_Ux_equals_y_for_x(U, y)
    n = length(y)
    x = zeros(n)
    
    # Backward substitution: solve from bottom to top
    for i = n:-1:1
        sum_val = 0.0
        for j = i+1:n
            sum_val += U[i,j] * x[j]
        end
        x[i] = (y[i] - sum_val) / U[i,i]
    end
    
    return x
end

"""
    solve_linear_system_LU(A, b)

Solve Ax = b using LU factorization.
This is the main driver function that uses get_LU_factors, 
solve_Ly_equals_b_for_y, and solve_Ux_equals_y_for_x.

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
    
    # Step 2: Solve Ly = b for y
    y = solve_Ly_equals_b_for_y(L, b)
    
    # Step 3: Solve Ux = y for x
    x = solve_Ux_equals_y_for_x(U, y)
    
    return x, L, U
end

# =============================================================================
# Cholesky Factorization Functions
# =============================================================================

"""
    get_cholesky_factor(A)

Compute Cholesky factorization of a symmetric positive-definite matrix A.
Returns L such that A = L*L'.

# Arguments
- `A::Matrix`: Symmetric positive-definite matrix

# Returns
- `L::Matrix`: Lower triangular matrix
"""
function get_cholesky_factor(A)
    n = size(A, 1)
    L = zeros(n, n)
    
    for i = 1:n
        # Diagonal elements
        sum_val = 0.0
        for k = 1:i-1
            sum_val += L[i,k]^2
        end
        L[i,i] = sqrt(A[i,i] - sum_val)
        
        # Off-diagonal elements (below diagonal)
        for j = i+1:n
            sum_val = 0.0
            for k = 1:i-1
                sum_val += L[j,k] * L[i,k]
            end
            L[j,i] = (A[j,i] - sum_val) / L[i,i]
        end
    end
    
    return L
end

"""
    solve_linear_system_cholesky(A, b)

Solve Ax = b using Cholesky factorization.
Assumes A is symmetric positive-definite.

# Arguments
- `A::Matrix`: Symmetric positive-definite coefficient matrix
- `b::Vector`: Right-hand side vector

# Returns
- `x::Vector`: Solution vector
- `L::Matrix`: Cholesky factor (lower triangular)
"""
function solve_linear_system_cholesky(A, b)
    # Step 1: Get Cholesky factor L where A = L*L'
    L = get_cholesky_factor(A)
    
    # Step 2: Solve Ly = b for y (forward substitution)
    n = length(b)
    y = zeros(n)
    for i = 1:n
        sum_val = 0.0
        for j = 1:i-1
            sum_val += L[i,j] * y[j]
        end
        y[i] = (b[i] - sum_val) / L[i,i]
    end
    
    # Step 3: Solve L'x = y for x (backward substitution with L')
    x = zeros(n)
    for i = n:-1:1
        sum_val = 0.0
        for j = i+1:n
            sum_val += L[j,i] * x[j]  # L' means we use L[j,i] instead of L[i,j]
        end
        x[i] = (y[i] - sum_val) / L[i,i]
    end
    
    return x, L
end

# =============================================================================
# Load testing utilities and test problems
# =============================================================================
include("src/test_utils.jl")

# =============================================================================
# Configuration: Control what runs when you execute this file
# =============================================================================

# Set to false to disable automatic testing
const RUN_TESTS = true

# Set to true to test full system solving (forward/backward substitution)
const TEST_FULL_SOLVER = true

# Set to true to also test Cholesky factorization
const TEST_CHOLESKY = true

# =============================================================================
# Main execution - Runs automatically (works with "Play" button!)
# =============================================================================

if RUN_TESTS
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "MIDTERM PROBLEM 3: LU and Cholesky Factorization", RESET)
    println(INFO, "=" ^ 80, RESET)
    
    # Define the midterm problem matrix and vector
    A = [6.25  -1.0   0.5;
        -1.0   5.0   2.12;
         0.5   2.12  3.6]
    b = [7.5, -8.68, -0.24]
    
    println(INFO, "\nGiven system Ax = b:", RESET)
    println("A = "); display(A)
    println("b = "); display(b)
    
    # =================================================================
    # LU Factorization Method
    # =================================================================
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "METHOD 1: LU Factorization", RESET)
    println(INFO, "=" ^ 80, RESET)
    
    x_lu, L, U = solve_linear_system_LU(A, b)
    
    println("\nL (lower triangular, 1's on diagonal):")
    display(L)
    println("\nU (upper triangular):")
    display(U)
    
    # Verify A = LU
    LU_product = L * U
    lu_error = maximum(abs.(A - LU_product))
    println("\nVerification: max|A - LU| = ", @sprintf("%.3e", lu_error))
    
    # Solution
    println(INFO, "\nSolution from LU factorization:", RESET)
    println("x = "); display(x_lu)
    
    # Verify Ax = b
    residual_lu = maximum(abs.(A * x_lu - b))
    println("Residual: max|Ax - b| = ", @sprintf("%.3e", residual_lu))
    
    # Compare with built-in
    x_builtin = A \ b
    diff_lu = maximum(abs.(x_lu - x_builtin))
    println("Difference from Julia's A\\b: ", @sprintf("%.3e", diff_lu))
    
    if diff_lu < 1e-10
        println(SUCCESS, "✓ LU solution matches built-in solver!", RESET)
    end
    
    # =================================================================
    # Cholesky Factorization Method
    # =================================================================
    if TEST_CHOLESKY
        println(INFO, "\n" * "=" ^ 80, RESET)
        println(INFO, "METHOD 2: Cholesky Factorization", RESET)
        println(INFO, "=" ^ 80, RESET)
        
        x_chol, L_chol = solve_linear_system_cholesky(A, b)
        
        println("\nL (Cholesky factor):")
        display(L_chol)
        
        # Verify A = L*L'
        chol_product = L_chol * L_chol'
        chol_error = maximum(abs.(A - chol_product))
        println("\nVerification: max|A - LL'| = ", @sprintf("%.3e", chol_error))
        
        # Solution
        println(INFO, "\nSolution from Cholesky factorization:", RESET)
        println("x = "); display(x_chol)
        
        # Verify Ax = b
        residual_chol = maximum(abs.(A * x_chol - b))
        println("Residual: max|Ax - b| = ", @sprintf("%.3e", residual_chol))
        
        # Compare with built-in
        diff_chol = maximum(abs.(x_chol - x_builtin))
        println("Difference from Julia's A\\b: ", @sprintf("%.3e", diff_chol))
        
        if diff_chol < 1e-10
            println(SUCCESS, "✓ Cholesky solution matches built-in solver!", RESET)
        end
        
        # Compare LU vs Cholesky
        diff_methods = maximum(abs.(x_lu - x_chol))
        println(INFO, "\nComparison: max|x_LU - x_Cholesky| = ", @sprintf("%.3e", diff_methods), RESET)
        if diff_methods < 1e-10
            println(SUCCESS, "✓ Both methods agree perfectly!", RESET)
        end
    end
    
    # =================================================================
    # Summary
    # =================================================================
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "SUMMARY", RESET)
    println(INFO, "=" ^ 80, RESET)
    println(@sprintf("%-25s %12.6f", "x₁ =", x_lu[1]))
    println(@sprintf("%-25s %12.6f", "x₂ =", x_lu[2]))
    println(@sprintf("%-25s %12.6f", "x₃ =", x_lu[3]))
    println(INFO, "=" ^ 80, RESET)
    
else
    # =================================================================
    # Manual/Interactive Mode
    # =================================================================
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "MANUAL MODE: Tests disabled", RESET)
    println(INFO, "=" ^ 80, RESET)
    println(INFO, "\nAvailable functions:", RESET)
    println(INFO, "  LU: get_LU_factors(A), solve_linear_system_LU(A, b)", RESET)
    println(INFO, "  Cholesky: get_cholesky_factor(A), solve_linear_system_cholesky(A, b)", RESET)
    println(INFO, "=" ^ 80, RESET)
end