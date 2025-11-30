# Testing and verification utilities for final project
# This file contains functions for testing numerical methods

"""
    print_header(title::String)

Print a formatted header for output sections.
"""
function print_header(title::String)
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, title, RESET)
    println(INFO, "=" ^ 80, RESET)
end

"""
    print_subheader(title::String)

Print a formatted subheader for output sections.
"""
function print_subheader(title::String)
    println(INFO, "\n" * "-" ^ 80, RESET)
    println(INFO, title, RESET)
    println(INFO, "-" ^ 80, RESET)
end

"""
    verify_sparse_recovery(x_true, x_recovered; tol=1e-6)

Verify that sparse signal recovery is accurate.

# Arguments
- `x_true::Vector`: True sparse signal
- `x_recovered::Vector`: Recovered signal from OMP
- `tol::Float64`: Tolerance for comparison (default: 1e-6)

# Returns
- `Bool`: true if recovery is successful within tolerance
"""
function verify_sparse_recovery(x_true, x_recovered; tol=1e-6)
    error = norm(x_true - x_recovered)
    rel_error = error / norm(x_true)
    
    println("\nRecovery Error Analysis:")
    println(@sprintf("  Absolute error: %.6e", error))
    println(@sprintf("  Relative error: %.6e", rel_error))
    
    if error < tol
        println(SUCCESS, "✓ Sparse recovery successful: error < $tol", RESET)
        return true
    else
        println(FAILURE, "✗ Sparse recovery failed: error = ", @sprintf("%.6e", error), RESET)
        return false
    end
end

"""
    verify_ode_solution(t_values, x_numeric, x_exact; name="Solution")

Verify numerical ODE solution against exact solution.

# Arguments
- `t_values::Vector`: Time points
- `x_numeric::Vector`: Numerical solution values
- `x_exact::Vector`: Exact solution values
- `name::String`: Name for the test case

# Returns
- `Dict`: Dictionary containing error metrics
"""
function verify_ode_solution(t_values, x_numeric, x_exact; name="Solution")
    errors = abs.(x_numeric .- x_exact)
    max_error = maximum(errors)
    mean_error = sum(errors) / length(errors)
    
    # Avoid division by zero
    rel_errors = errors ./ (abs.(x_exact) .+ 1e-15)
    max_rel_error = maximum(rel_errors)
    
    println("\n$name Error Analysis:")
    println(@sprintf("  Maximum absolute error: %.6e", max_error))
    println(@sprintf("  Mean absolute error:    %.6e", mean_error))
    println(@sprintf("  Maximum relative error: %.6e", max_rel_error))
    
    if max_error < 1e-4
        println(SUCCESS, "✓ Excellent accuracy: max error < 1e-4", RESET)
    elseif max_error < 1e-2
        println(INFO, "✓ Good accuracy: max error < 1e-2", RESET)
    else
        println(WARNING, "⚠ Moderate accuracy: max error = ", @sprintf("%.6e", max_error), RESET)
    end
    
    return Dict(
        "max_error" => max_error,
        "mean_error" => mean_error,
        "max_rel_error" => max_rel_error
    )
end

"""
    compare_solutions(name1, sol1, name2, sol2; show_all=false)

Compare two solution vectors and display differences.

# Arguments
- `name1::String`: Name of first solution
- `sol1::Vector`: First solution vector
- `name2::String`: Name of second solution
- `sol2::Vector`: Second solution vector
- `show_all::Bool`: Whether to show all entries or just first/last few
"""
function compare_solutions(name1, sol1, name2, sol2; show_all=false)
    diff = sol1 - sol2
    max_diff = maximum(abs.(diff))
    
    println("\nSolution Comparison:")
    println(@sprintf("%-15s %-15s %-15s", name1, name2, "Difference"))
    println("-" ^ 50)
    
    n = length(sol1)
    indices = show_all ? (1:n) : [1:min(3,n); max(4, n-2):n]
    
    for i in indices
        if i <= n
            println(@sprintf("%-15.6f %-15.6f %-15.6e", sol1[i], sol2[i], diff[i]))
        end
        if !show_all && i == 3 && n > 6
            println("  ⋮                ⋮                ⋮")
        end
    end
    
    println("-" ^ 50)
    println(@sprintf("Maximum difference: %.6e", max_diff))
    
    return max_diff
end

"""
    print_matrix_info(A; name="Matrix")

Print useful information about a matrix.

# Arguments
- `A::Matrix`: Matrix to analyze
- `name::String`: Name for the matrix
"""
function print_matrix_info(A; name="Matrix")
    m, n = size(A)
    println("\n$name Properties:")
    println(@sprintf("  Dimensions: %d × %d", m, n))
    println(@sprintf("  Rank: %d", rank(A)))
    
    if m == n
        println(@sprintf("  Condition number: %.4e", cond(A)))
        println(@sprintf("  Determinant: %.4e", det(A)))
    else
        println(@sprintf("  Condition number: %.4e", cond(A)))
    end
    
    # Singular values
    σ = svd(A).S
    println(@sprintf("  Largest singular value: %.4e", σ[1]))
    println(@sprintf("  Smallest singular value: %.4e", σ[end]))
end

"""
    save_plot_safe(p, filename; output_dir="../tex/figures")

Safely save a plot, creating directory if needed.

# Arguments
- `p`: Plot object
- `filename::String`: Output filename
- `output_dir::String`: Output directory path
"""
function save_plot_safe(p, filename; output_dir="../tex/figures")
    if !isdir(output_dir)
        mkpath(output_dir)
        println(INFO, "✓ Created directory: $output_dir", RESET)
    end
    
    filepath = joinpath(output_dir, filename)
    savefig(p, filepath)
    println(SUCCESS, "✓ Plot saved to: $filepath", RESET)
end

"""
    format_table_row(values...; formats=nothing)

Format a table row with specified formats for each column.

# Arguments
- `values...`: Values to format
- `formats`: Array of format strings (default: auto-detect)
"""
function format_table_row(values...; formats=nothing)
    if formats === nothing
        # Auto-detect formats
        formats = [v isa AbstractFloat ? "%.6f" : "%s" for v in values]
    end
    
    row = ""
    for (val, fmt) in zip(values, formats)
        row *= @sprintf("%15s ", @sprintf(fmt, val))
    end
    return row
end

println(INFO, "✓ Test utilities loaded", RESET)
