# CPTS530 Final Project - Problem 1: Orthogonal Matching Pursuit (OMP)
# Author: Aryan Ritwajeet Jha
# Date: December 2025

using LinearAlgebra
using Printf

# Color constants for terminal output
const SUCCESS = "\e[32m"  # Green
const FAILURE = "\e[31m"  # Red
const INFO = "\e[36m"     # Cyan
const WARNING = "\e[33m"  # Yellow
const RESET = "\e[0m"     # Reset color

println(INFO, "="^80, RESET)
println(INFO, "CPTS530 Final Project - Problem 1: Orthogonal Matching Pursuit", RESET)
println(INFO, "="^80, RESET)

# =================================================================
# Part 0: Initialize matrices A and B
# =================================================================

println(INFO, "\n📊 Initializing Matrices", RESET)
println(INFO, "-"^80, RESET)

# Matrix A (3×3) from problem statement
A = [
    1.0      -1/2      -1/2;
    0.0   sqrt(3)/2  -sqrt(3)/2;
    1.0      3.0       3.0
]

# Matrix B (2×3) - first two rows of A
B = A[1:2, :]

println("\nMatrix A (3×3):")
display(A)

println("\n\nMatrix B (2×3):")
display(B)

# Verify dimensions
println(INFO, "\n✓ Matrix A dimensions: ", size(A), RESET)
println(INFO, "✓ Matrix B dimensions: ", size(B), RESET)

# Compute and display singular value decompositions
println(INFO, "\n\n🔍 Computing SVD for Matrix A", RESET)
println(INFO, "-"^80, RESET)

U_A, Σ_A, V_A = svd(A)
println("\nSingular values of A: ", Σ_A)
println("Rank of A: ", rank(A))

println(INFO, "\n\n🔍 Computing SVD for Matrix B", RESET)
println(INFO, "-"^80, RESET)

U_B, Σ_B, V_B = svd(B)
println("\nSingular values of B: ", Σ_B)
println("Rank of B: ", rank(B))

# =================================================================
# Part 1: Implement Orthogonal Matching Pursuit (OMP)
# =================================================================

"""
    hard_threshold(x, s)

Hard thresholding operator H_s(x): sets all entries of x to zero 
except for the s largest (in absolute value) entries.

# Arguments
- `x::Vector`: Input vector
- `s::Int`: Number of largest entries to keep

# Returns
- `Vector`: Thresholded vector
"""
function hard_threshold(x::Vector, s::Int)
    n = length(x)
    if s >= n
        return copy(x)
    end
    
    # Get indices sorted by absolute value (descending)
    sorted_indices = sortperm(abs.(x), rev=true)
    
    # Create thresholded vector
    x_thresholded = zeros(n)
    x_thresholded[sorted_indices[1:s]] = x[sorted_indices[1:s]]
    
    return x_thresholded
end

"""
    orthogonal_matching_pursuit(A, b, s, max_iterations=100)

Orthogonal Matching Pursuit algorithm for sparse signal recovery.

# Arguments
- `A::Matrix`: Measurement matrix (m×n)
- `b::Vector`: Observation vector (m×1)
- `s::Int`: Sparsity level (number of non-zero entries to recover)
- `max_iterations::Int`: Maximum number of iterations (default: 100)

# Returns
- `x::Vector`: Recovered sparse signal
- `history::Dict`: Dictionary containing iteration history
"""
function orthogonal_matching_pursuit(A::Matrix, b::Vector, s::Int; max_iterations::Int=100)
    m, n = size(A)
    
    # Initialize
    x = zeros(n)              # Solution vector
    residual = copy(b)        # Current residual
    support = Int[]           # Indices of selected columns
    
    # History tracking
    history = Dict(
        "iterations" => Int[],
        "residuals" => Float64[],
        "support" => Vector{Int}[]
    )
    
    println(INFO, "\n\n🎯 Running Orthogonal Matching Pursuit", RESET)
    println(INFO, "-"^80, RESET)
    println("Sparsity level s = ", s)
    println("Matrix dimensions: ", size(A))
    println("Starting residual norm: ", @sprintf("%.6e", norm(residual)))
    
    # OMP iterations
    for iter in 1:min(s, max_iterations)
        # Step 1: Find column with largest correlation with residual
        correlations = abs.(A' * residual)
        max_idx = argmax(correlations)
        
        # Step 2: Add to support if not already present
        if !(max_idx in support)
            push!(support, max_idx)
        end
        
        # Step 3: Solve least squares problem on support
        A_support = A[:, support]
        x_support = pinv(A_support) * b  # Use pseudo-inverse
        
        # Step 4: Update solution
        x = zeros(n)
        x[support] = x_support
        
        # Step 5: Update residual
        residual = b - A * x
        
        # Record history
        push!(history["iterations"], iter)
        push!(history["residuals"], norm(residual))
        push!(history["support"], copy(support))
        
        println(@sprintf("\nIteration %2d: Selected column %d, residual = %.6e", 
                iter, max_idx, norm(residual)))
        
        # Check convergence
        if norm(residual) < 1e-10
            println(SUCCESS, "\n✓ Converged! Residual below threshold.", RESET)
            break
        end
    end
    
    println(INFO, "\n" * "="^80, RESET)
    println(SUCCESS, "OMP Complete!", RESET)
    println("Final support set: ", support)
    println("Final residual norm: ", @sprintf("%.6e", norm(residual)))
    println("Solution vector x:")
    display(x)
    
    return x, history
end

# =================================================================
# Part 2: Test with matrix A and e₁ = (1, 0, 0)ᵀ
# =================================================================

println(INFO, "\n\n" * "="^80, RESET)
println(INFO, "PART 2: Testing OMP with Matrix A", RESET)
println(INFO, "="^80, RESET)

e1 = [1.0, 0.0, 0.0]
b_A = A * e1
s = 1  # Sparsity level

println("\nTarget vector e₁ = ", e1)
println("Observation vector b = A*e₁ = ", b_A)

x_recovered_A, history_A = orthogonal_matching_pursuit(A, b_A, s)

println(INFO, "\n\n📊 Comparison Results for Matrix A:", RESET)
println(INFO, "-"^80, RESET)
println("Target:    ", e1)
println("Recovered: ", x_recovered_A)
println("Error:     ", norm(e1 - x_recovered_A))

if norm(e1 - x_recovered_A) < 1e-6
    println(SUCCESS, "✓ Excellent recovery! Error < 1e-6", RESET)
else
    println(WARNING, "⚠ Recovery not perfect. Error = ", norm(e1 - x_recovered_A), RESET)
end

# =================================================================
# Part 3: Test with matrix B
# =================================================================

println(INFO, "\n\n" * "="^80, RESET)
println(INFO, "PART 3: Testing OMP with Matrix B", RESET)
println(INFO, "="^80, RESET)

b_B = B * e1
println("\nTarget vector e₁ = ", e1)
println("Observation vector b = B*e₁ = ", b_B)

x_recovered_B, history_B = orthogonal_matching_pursuit(B, b_B, s)

println(INFO, "\n\n📊 Comparison Results for Matrix B:", RESET)
println(INFO, "-"^80, RESET)
println("Target:    ", e1)
println("Recovered: ", x_recovered_B)
println("Error:     ", norm(e1 - x_recovered_B))

if norm(e1 - x_recovered_B) < 1e-6
    println(SUCCESS, "✓ Excellent recovery! Error < 1e-6", RESET)
else
    println(WARNING, "⚠ Recovery not perfect. Error = ", norm(e1 - x_recovered_B), RESET)
end

# =================================================================
# Part 4: Analysis and Comparison
# =================================================================

println(INFO, "\n\n" * "="^80, RESET)
println(INFO, "PART 4: Analysis and Comparison", RESET)
println(INFO, "="^80, RESET)

println("\n🔍 Matrix Properties:")
println("-"^80)
println("Matrix A:")
println("  - Dimensions: ", size(A))
println("  - Rank: ", rank(A))
println("  - Condition number: ", @sprintf("%.4f", cond(A)))

println("\nMatrix B:")
println("  - Dimensions: ", size(B))
println("  - Rank: ", rank(B))
println("  - Condition number: ", @sprintf("%.4f", cond(B)))

println("\n\n📈 Recovery Performance:")
println("-"^80)
println(@sprintf("Matrix A - Recovery error: %.6e", norm(e1 - x_recovered_A)))
println(@sprintf("Matrix B - Recovery error: %.6e", norm(e1 - x_recovered_B)))

println("\n\n💡 Connection to Compressed Sensing:")
println("-"^80)
println("""
Matrix A (3×3):
- Square matrix with full rank
- Provides m=3 measurements for n=3 unknowns
- Standard (non-compressed) scenario
- Perfect recovery expected for 1-sparse signal

Matrix B (2×3):
- Underdetermined system (fewer measurements than unknowns)
- Provides m=2 measurements for n=3 unknowns
- Compressed sensing scenario
- Recovery depends on sparsity and matrix properties (RIP, coherence)
- Success demonstrates sparse recovery from incomplete measurements
""")

println(INFO, "\n" * "="^80, RESET)
println(SUCCESS, "Problem 1 Analysis Complete! ✓", RESET)
println(INFO, "="^80, RESET)
