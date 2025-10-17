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

# =============================================================================
# Load testing utilities and test problems
# =============================================================================
include("src/test_utils.jl")

# =============================================================================
# Configuration: Control what runs when you execute this file
# =============================================================================

# Set to false to disable automatic testing
const RUN_TESTS = true

# Set to false if you just want to test LU factorization (not full system solving)
const TEST_FULL_SOLVER = false

# =============================================================================
# Main execution - Runs automatically (works with "Play" button!)
# =============================================================================

if RUN_TESTS
        println(INFO, "\n" * "=" ^ 80, RESET)
        println(INFO, "MIDTERM PROBLEM 3: LU FACTORIZATION TESTS", RESET)
        println(INFO, "=" ^ 80, RESET)
        
        if TEST_FULL_SOLVER
            # =================================================================
            # Full System Solving Tests (when forward/backward substitution ready)
            # =================================================================
            println(INFO, "\n" * "=" ^ 80, RESET)
            println(INFO, "FULL SYSTEM SOLVING TESTS", RESET)
            println(INFO, "=" ^ 80, RESET)
            
            test_problem_1()      # 3×3 midterm system
            test_problem_2()      # 2×2 simple system
            test_problem_identity()  # Identity matrix
            test_problem_diagonal()  # Diagonal matrix
            
        else
            # =================================================================
            # LU Factorization Only Tests
            # =================================================================
            
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
            
            # Test Case 2: ACTUAL MIDTERM PROBLEM 3 (3×3)
            println(INFO, "\n\nTEST 2: ACTUAL Midterm Problem 3", RESET)
            A2 = [6.25  -1.0   0.5;
                 -1.0   5.0   2.12;
                  0.5   2.12  3.6]
            test_LU_only(A2, name="Midterm Problem 3 - Actual Matrix")
            
            println("\n", INFO, "Note: This is the matrix from your midterm problem!", RESET)
            
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
            println(INFO, "  - Finally set TEST_FULL_SOLVER = true", RESET)
            println(INFO, "=" ^ 80, RESET)
        end
        
    else
        # =================================================================
        # Manual/Interactive Mode: No automatic tests
        # =================================================================
        println(INFO, "\n" * "=" ^ 80, RESET)
        println(INFO, "MANUAL MODE: Tests disabled (RUN_TESTS = false)", RESET)
        println(INFO, "=" ^ 80, RESET)
        println(INFO, "\nYou can now use functions interactively:", RESET)
        println(INFO, "  - get_LU_factors(A)", RESET)
        println(INFO, "  - solve_Ly_is_equal_to_b(L, b)", RESET)
        println(INFO, "  - solve_Ux_is_equal_to_y(U, y)", RESET)
        println(INFO, "  - solve_linear_system_LU(A, b)", RESET)
        println(INFO, "\nOr call specific tests:", RESET)
        println(INFO, "  - test_LU_only(A, name=\"My Test\")", RESET)
        println(INFO, "  - test_problem_1(), test_problem_2(), etc.", RESET)
        println(INFO, "=" ^ 80, RESET)
        
        # Example: Uncomment to test a specific matrix
        # A = [2.0 1.0 1.0; 4.0 -6.0 0.0; -2.0 7.0 2.0]
        # L, U = get_LU_factors(A)
        # println("\nL = "); display(L)
        # println("\nU = "); display(U)
end
