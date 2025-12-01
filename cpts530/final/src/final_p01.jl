# CPTS530 Final Project - Problem 1: Orthogonal Matching Pursuit (OMP)
# Author: Aryan Ritwajeet Jha
# Date: December 2025

using LinearAlgebra
using Printf

# Load environment and utilities
# include("preamble.jl")
# include("test_utils.jl")

println("="^80)
println("CPTS530 Final Project - Problem 1: Orthogonal Matching Pursuit")
println("="^80)

# =================================================================
# Part 0: Initialize matrices A and B
# =================================================================

println("\n📊 Initializing Matrices")
println("-"^80)

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
println( "\n✓ Matrix A dimensions: ", size(A))
println( "✓ Matrix B dimensions: ", size(B))

# Compute and display singular value decompositions
println( "\n\n🔍 Computing SVD for Matrix A")
println( "-"^80)

U_A, Σ_A, V_A = svd(A)
println("\nSingular values of A: ", Σ_A)
println("Rank of A: ", rank(A))

println( "\n\n🔍 Computing SVD for Matrix B")
println( "-"^80)

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
    
    println( "\n\n🎯 Running Orthogonal Matching Pursuit")
    println( "-"^80)
    println("Sparsity level s = ", s)
    println("Matrix dimensions: ", size(A))
    println("Starting residual norm: ", @sprintf("%.6e", norm(residual)))
    
    # OMP iterations
    for iter in 1:min(s, max_iterations)
        # Step 1: Compute correlations g_k = A^T * r_k
        g_k = A' * residual
        
        # Step 2: Apply hard thresholding H_s(g_k) and find support
        g_k_thresholded = hard_threshold(g_k, s)
        new_indices = findall(x -> abs(x) > 1e-14, g_k_thresholded)
        
        # Step 3: Update support S_{k+1} = S_k ∪ {j_{k+1}}
        for idx in new_indices
            if !(idx in support)
                push!(support, idx)
            end
        end
        
        # Step 4: Solve least squares problem on support
        # (x_{k+1})_{S_{k+1}} = A^+_{S_{k+1}} * b
        # (x_{k+1})_{S^c_{k+1}} = 0
        A_support = A[:, support]
        x_support = pinv(A_support) * b  # Use pseudo-inverse
        
        # Update solution
        x = zeros(n)
        x[support] = x_support
        
        # Step 5: Update residual r_{k+1} = b - A * x_{k+1}
        residual = b - A * x
        
        # Record history
        push!(history["iterations"], iter)
        push!(history["residuals"], norm(residual))
        push!(history["support"], copy(support))
        
        println(@sprintf("\nIteration %2d: Support size = %d, residual = %.6e", 
                iter, length(support), norm(residual)))
        println("  Current support: ", support)
        
        # Check convergence
        if norm(residual) < 1e-10
            println( "\n✓ Converged! Residual below threshold.")
            break
        end
        
        # Stop if we've already selected s indices
        if length(support) >= s
            println( "\n✓ Reached desired sparsity level s = $s")
            break
        end
    end
    
    println( "\n" * "="^80)
    println( "OMP Complete!")
    println("Final support set: ", support)
    println("Final residual norm: ", @sprintf("%.6e", norm(residual)))
    println("Solution vector x:")
    display(x)
    
    return x, history
end

# =================================================================
# Part 2: Test with matrix A and e₁ = (1, 0, 0)ᵀ
# =================================================================

println( "\n\n" * "="^80)
println( "PART 2: Testing OMP with Matrix A")
println( "="^80)

e1 = [1.0, 0.0, 0.0]
b_A = A * e1
s = 1  # Sparsity level

println("\nTarget vector e₁ = ", e1)
println("Observation vector b = A*e₁ = ", b_A)

x_recovered_A, history_A = orthogonal_matching_pursuit(A, b_A, s)

println( "\n\n📊 Comparison Results for Matrix A:")
println( "-"^80)
println("Target:    ", e1)
println("Recovered: ", x_recovered_A)
println("Error:     ", norm(e1 - x_recovered_A))

if norm(e1 - x_recovered_A) < 1e-6
    println( "✓ Excellent recovery! Error < 1e-6")
else
    println( "⚠ Recovery not perfect. Error = ", norm(e1 - x_recovered_A))
end

# =================================================================
# Part 3: Test with matrix B
# =================================================================

println( "\n\n" * "="^80)
println( "PART 3: Testing OMP with Matrix B")
println( "="^80)

b_B = B * e1
println("\nTarget vector e₁ = ", e1)
println("Observation vector b = B*e₁ = ", b_B)

x_recovered_B, history_B = orthogonal_matching_pursuit(B, b_B, s)

println( "\n\n📊 Comparison Results for Matrix B:")
println( "-"^80)
println("Target:    ", e1)
println("Recovered: ", x_recovered_B)
println("Error:     ", norm(e1 - x_recovered_B))

if norm(e1 - x_recovered_B) < 1e-6
    println( "✓ Excellent recovery! Error < 1e-6")
else
    println( "⚠ Recovery not perfect. Error = ", norm(e1 - x_recovered_B))
end

# =================================================================
# Part 4: Analysis and Comparison
# =================================================================

println( "\n\n" * "="^80)
println( "PART 4: Analysis and Comparison")
println( "="^80)

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

println( "\n" * "="^80)
println( "Problem 1 Analysis Complete! ✓")
println( "="^80)
