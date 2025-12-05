# CPTS530 Final Project - Problem 1: Orthogonal Matching Pursuit (OMP)
# Author: Aryan Ritwajeet Jha
# Date: December 2025

# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using LinearAlgebra
using Printf
using Crayons

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
    orthogonal_matching_pursuit(A, b, s, max_iterations=100; verbose=false)

Orthogonal Matching Pursuit algorithm for sparse signal recovery.

# Arguments
- `A::Matrix`: Measurement matrix (m×n)
- `b::Vector`: Observation vector (m×1)
- `s::Int`: Sparsity level (number of non-zero entries to recover)
- `max_iterations::Int`: Maximum number of iterations (default: 100)
- `verbose::Bool`: Print detailed iteration info (default: false)

# Returns
- `x::Vector`: Recovered sparse signal
- `history::Dict`: Dictionary containing iteration history
"""
function orthogonal_matching_pursuit(A::Matrix, b::Vector, s::Int; max_iterations::Int=100, verbose::Bool=false)
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
    
    # OMP iterations
    for iter in 1:min(s, max_iterations)
        # Step 1: Compute correlations g_k = A^T * r_k
        g_k = A' * residual
        
        # Step 2: Find the index with maximum absolute correlation
        # (that's not already in the support)
        max_corr = -Inf
        best_idx = 0
        for idx in 1:n
            if !(idx in support) && abs(g_k[idx]) > max_corr
                max_corr = abs(g_k[idx])
                best_idx = idx
            end
        end
        
        # Step 3: Add the best index to support
        if best_idx > 0
            push!(support, best_idx)
        else
            break  # No more columns to add
        end
        
        # Step 4: Solve least squares problem on support
        # (x_{k+1})_{S_{k+1}} = A^+_{S_{k+1}} * b
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
        
        # Check convergence
        if norm(residual) < 1e-10
            break
        end
        
        # Stop if we've already selected s indices
        if length(support) >= s
            break
        end
    end
    
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
s = 3  # Sparsity level (increased from 1 to 3)

println("\nTarget vector e₁ = ", e1)
println("Observation vector b = A*e₁ = ", b_A)

x_recovered_A, history_A = orthogonal_matching_pursuit(A, b_A, s)

println( "\n\n📊 Comparison Results for Matrix A:")
println( "-"^80)
println("Target:    ", e1)
# =================================================================
# Part 2: Test with matrix A for s = 1, 2, ..., rank(A)
# =================================================================

println( "\n\n" * "="^80)
println( "PART 2: Testing OMP with Matrix A")
println( "="^80)

e1 = [1.0, 0.0, 0.0]
b_A = A * e1
rank_A = rank(A)

println(Crayon(foreground=:cyan, bold=true), "\n🎯 Target: e₁ = $e1")
println(Crayon(reset=true), "   Observation: b = A*e₁ = $b_A")
println("   Rank of A = $rank_A\n")

# Store solutions for Matrix A
xA = Vector{Vector{Float64}}()

for s in 1:rank_A
    x_recovered, history = orthogonal_matching_pursuit(A, b_A, s)
    push!(xA, x_recovered)
    error = norm(e1 - x_recovered)
    
    if error < 1e-6
        println(Crayon(foreground=:green, bold=true), "✓ s=$s: ", Crayon(reset=true), 
                "x = $x_recovered, error = ", @sprintf("%.2e", error))
    else
        println(Crayon(foreground=:yellow, bold=true), "⚠ s=$s: ", Crayon(reset=true),
                "x = $x_recovered, error = ", @sprintf("%.2e", error))
    end
end

# =================================================================
# Part 3: Test with matrix B for s = 1, 2, ..., rank(B)
# =================================================================

println( "\n\n" * "="^80)
println( "PART 3: Testing OMP with Matrix B")
println( "="^80)

b_B = B * e1
rank_B = rank(B)

println(Crayon(foreground=:cyan, bold=true), "\n🎯 Target: e₁ = $e1")
println(Crayon(reset=true), "   Observation: b = B*e₁ = $b_B")
println("   Rank of B = $rank_B\n")

# Store solutions for Matrix B
xB = Vector{Vector{Float64}}()

for s in 1:rank_B
    x_recovered, history = orthogonal_matching_pursuit(B, b_B, s)
    push!(xB, x_recovered)
    error = norm(e1 - x_recovered)
    
    if error < 1e-6
        println(Crayon(foreground=:green, bold=true), "✓ s=$s: ", Crayon(reset=true),
                "x = $x_recovered, error = ", @sprintf("%.2e", error))
    else
        println(Crayon(foreground=:yellow, bold=true), "⚠ s=$s: ", Crayon(reset=true),
                "x = $x_recovered, error = ", @sprintf("%.2e", error))
    end
end

# =================================================================
# Summary
# =================================================================

println( "\n\n" * "="^80)
println(Crayon(foreground=:magenta, bold=true), "SUMMARY")
println(Crayon(reset=true), "="^80)

println("\nMatrix A ($(size(A)[1])×$(size(A)[2]), rank=$(rank(A))):")
println("  Square, full rank → Standard recovery scenario")

println("\nMatrix B ($(size(B)[1])×$(size(B)[2]), rank=$(rank(B))):")
println("  Underdetermined → Compressed sensing scenario")

println( "\n" * "="^80)
println(Crayon(foreground=:green, bold=true), "Problem 1 Complete! ✓")
println(Crayon(reset=true), "="^80)

# =================================================================
# Part 4: Automated Analysis and Results Table
# =================================================================

println("\n\n" * "="^80)
println(Crayon(foreground=:magenta, bold=true), "DETAILED ANALYSIS")
println(Crayon(reset=true), "="^80)

# Compute residuals for all solutions
println("\n📊 Observation Vector Analysis:")
println("-"^80)
println("Matrix A:")
println("  b_A = $b_A")
println("  ‖b_A‖₀ (sparsity) = ", count(x -> abs(x) > 1e-10, b_A))
println("  ‖b_A‖₂ (norm) = ", @sprintf("%.4f", norm(b_A)))

println("\nMatrix B:")
println("  b_B = $b_B")
println("  ‖b_B‖₀ (sparsity) = ", count(x -> abs(x) > 1e-10, b_B))
println("  ‖b_B‖₂ (norm) = ", @sprintf("%.4f", norm(b_B)))

# Create results table
println("\n\n📋 Results Table:")
println("-"^80)
println(Crayon(bold=true), @sprintf("%-10s %-4s %-15s %-15s %-12s %-12s", 
        "Matrix", "s", "‖x - e₁‖₂", "‖Ax - b‖₂", "Recovery", "Feasible"))
println(Crayon(reset=true), "-"^80)

# Matrix A results
for (i, s) in enumerate(1:rank_A)
    error_target = norm(e1 - xA[i])
    error_obs = norm(A * xA[i] - b_A)
    recovery = error_target < 1e-6 ? "✓" : "✗"
    feasible = error_obs < 1e-6 ? "✓" : "✗"
    
    color = error_target < 1e-6 ? Crayon(foreground=:green) : Crayon(foreground=:yellow)
    println(color, @sprintf("%-10s %-4d %-15s %-15s %-12s %-12s",
            "A (3×3)", s, 
            @sprintf("%.2e", error_target),
            @sprintf("%.2e", error_obs),
            recovery, feasible))
end

# Matrix B results
for (i, s) in enumerate(1:rank_B)
    error_target = norm(e1 - xB[i])
    error_obs = norm(B * xB[i] - b_B)
    recovery = error_target < 1e-6 ? "✓" : "✗"
    feasible = error_obs < 1e-6 ? "✓" : "✗"
    
    color = error_target < 1e-6 ? Crayon(foreground=:green) : Crayon(foreground=:yellow)
    println(color, @sprintf("%-10s %-4d %-15s %-15s %-12s %-12s",
            "B (2×3)", s,
            @sprintf("%.2e", error_target),
            @sprintf("%.2e", error_obs),
            recovery, feasible))
end

println(Crayon(reset=true), "-"^80)

# Key observations
println("\n\n💡 Key Observations:")
println("-"^80)
println(Crayon(foreground=:cyan, bold=true), "Feasibility vs Recovery:")
println(Crayon(reset=true), "• Matrix A (s=1): Neither feasible nor recovers e₁")
println("• Matrix A (s≥2): Both feasible and recovers e₁")
println("• Matrix B (s≥1): Both feasible and recovers e₁")
println()
println(Crayon(foreground=:cyan, bold=true), "Critical Insight:")
println(Crayon(reset=true), "Success depends on the sparsity of b in the column space of the")
println("measurement matrix, NOT the sparsity of the original signal x.")
println()
println("• Matrix A needs s=2: b_A = [1,0,1] requires 2 columns for representation")
println("• Matrix B needs s=1: b_B = [1,0] requires only 1 column for representation")
println()
println(Crayon(foreground=:cyan, bold=true), "Compressed Sensing Advantage:")
println(Crayon(reset=true), "Matrix B (2×3, underdetermined) recovers the 1-sparse signal with fewer")
println("measurements than Matrix A (3×3) because B is inherently better designed")
println("to capture 1-sparse signals—by removing redundant measurements, it")
println("efficiently represents sparse observations with minimal dimensions.")

# Make solutions and observations available in global scope
println("\n\n💾 Variables available:")
println("   Observations: b_A = $b_A, b_B = $b_B")
println("   Solutions: xA[1], xA[2], xA[3], xB[1], xB[2]")
println("   Target: e1 = $e1")
