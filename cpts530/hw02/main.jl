# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Parameters
using Crayons
using Printf

# Print like println when no verbose flag is given
myprintln(args...) = println(args...)

# Only print when verbose==true
myprintln(verbose::Bool, args...) = verbose && println(args...)

# IO-aware variants (optional but nice to have)
myprintln(io::IO, args...) = println(io, args...)
myprintln(io::IO, verbose::Bool, args...) = verbose && println(io, args...)


function bisection_search(pr; verbose=false)
    @unpack a, b, M, ϵ, δ, f = pr;
    itr = 1
    shouldStop = false
    u = f(a); v = f(b); e = b - a;
    sgn_f_a = sign(u); sgn_f_b = sign(v);
    if sgn_f_a * sgn_f_b > 0
        shouldStop = true
        error("f(a) and f(b) must have opposite signs for bisection_search to work (using MVT).")
        return nothing, nothing
    end

    while !shouldStop
        e = e / 2
        c = a + e
        myprintln(verbose, "itr=$itr, a=$a, b=$b, c=$c, f(c)=$(f(c)), |b-a|=$(2*abs(e))")
        w = f(c)
        if abs(w) < ϵ
            shouldStop = true
            println("Root found at c = $c with f(c) = $w")
            return c, w
        end
        if abs(e) < δ
            shouldStop = true
            @error("Interval sufficiently small: |b - a| = $(2*abs(e)) < δ = $δ")
            return nothing, nothing
        end
        sgn_f_c = sign(w)
        if sgn_f_a * sgn_f_c < 0 # root is in [a, c]
            b = c
            v = w
            sgn_f_b = sgn_f_c
        elseif sgn_f_b * sgn_f_c < 0 # root is in [c, b]
            a = c
            u = w
            sgn_f_a = sgn_f_c
        else # w == 0, found exact root?
            @error("floc")
            shouldStop = true
            return nothing, nothing
        end
        itr += 1
        if itr > M
            shouldStop = true
            @error("Exceeded maximum iterations M = $M")
            return nothing, nothing
        end
    end
end

function ridders_search(pr; verbose=false)
    @unpack a, b, M, ϵ, δ, f = pr;
    itr = 1
    shouldStop = false
    
    # Initial bracketing interval [x0, x2]
    x0 = a
    x2 = b
    f0 = f(x0)
    f2 = f(x2)
    
    sgn_f0 = sign(f0)
    sgn_f2 = sign(f2)
    
    if sgn_f0 * sgn_f2 > 0
        shouldStop = true
        error("f(x0) and f(x2) must have opposite signs for ridders_search to work.")
        return nothing, nothing
    end
    
    while !shouldStop
        # Compute midpoint x1
        x1 = (x0 + x2) / 2
        f1 = f(x1)
        
        # Compute distance d
        d = x2 - x0
        
        myprintln(verbose, "itr=$itr, x0=$x0, x2=$x2, x1=$x1, f0=$f0, f1=$f1, f2=$f2, d=$d")
        
        # Check convergence on interval size
        if abs(d) < δ
            shouldStop = true
            @error("Interval sufficiently small: |x2 - x0| = $d < δ = $δ")
            return nothing, nothing
        end
        
        # Compute parameter a for h(x) = f(x)*e^(a*x)
        # a = (2/d) * ln((f1 + sgn(f2)*sqrt(f1^2 - f2*f0)) / f2)
        discriminant = f1^2 - f2 * f0
        
        if discriminant < 0
            shouldStop = true
            @error("Negative discriminant: f1^2 - f2*f0 = $discriminant < 0")
            return nothing, nothing
        end
        
        a_param = (2 / d) * log((f1 + sign(f2) * sqrt(discriminant)) / f2)
        
        # Compute h(x) values
        h0 = f0 * exp(a_param * x0)
        h1 = f1 * exp(a_param * x1)
        h2 = f2 * exp(a_param * x2)
        
        myprintln(verbose, "    a_param=$a_param, h0=$h0, h1=$h1, h2=$h2")
        
        # Compute x3 (x-intercept of line through (x0, h0) and (x2, h2))
        # x3 = (x1*h2 - x2*h1) / (h2 - h1)
        # Alternative form: x3 = x1 - h1*(x1 - x2)/(h1 - h2)
        if abs(h2 - h1) < 1e-14
            shouldStop = true
            @error("h2 - h1 too small: cannot compute x3")
            return nothing, nothing
        end
        
        x3 = (x1 * h2 - x2 * h1) / (h2 - h1)
        f3 = f(x3)
        
        myprintln(verbose, "    x3=$x3, f3=$f3")
        
        # Check convergence on function value
        if abs(f3) < ϵ
            shouldStop = true
            println("Root found at x3 = $x3 with f(x3) = $f3")
            return x3, f3
        end
        
        # Determine next bracketing interval
        sgn_f1 = sign(f1)
        sgn_f3 = sign(f3)
        
        if sgn_f1 * sgn_f3 < 0
            # Root is in [x1, x3] or [x3, x1]
            if x1 < x3
                x0 = x1
                x2 = x3
                f0 = f1
                f2 = f3
            else
                x0 = x3
                x2 = x1
                f0 = f3
                f2 = f1
            end
        elseif sgn_f0 * sgn_f3 < 0
            # Root is in [x0, x3]
            x2 = x3
            f2 = f3
        elseif sgn_f2 * sgn_f3 < 0
            # Root is in [x3, x2]
            x0 = x3
            f0 = f3
        else
            shouldStop = true
            @error("Cannot determine next bracketing interval")
            return nothing, nothing
        end
        
        sgn_f0 = sign(f0)
        sgn_f2 = sign(f2)
        
        itr += 1
        if itr > M
            shouldStop = true
            @error("Exceeded maximum iterations M = $M")
            return nothing, nothing
        end
    end
end


# Root = None in the interval [1,2] (tan has asymptote at π/2 ≈ 1.5708)
pr = Dict(
    :a => 1,
    :b => 2,
    :M => 10000,
    :ϵ => 1e-8,
    :δ => 1e-8,
    :f => (x -> x - tan(x))
)

# Root ≈ 0.7390851332
pr1 = Dict(
    :a => 0.0,
    :b => 1.0,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> cos(x) - x)
)

# Root ≈ 0.5671432904
pr2 = Dict(
    :a => 0.0,
    :b => 1.0,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> exp(-x) - x)
)

# Root ≈ 1.3247179572
pr3 = Dict(
    :a => 1.0,
    :b => 2.0,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-6,
    :f => (x -> x^3 - x - 1)
)

# Root ≈ 1.8954942670
pr4 = Dict(
    :a => 1.5,
    :b => 2.5,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> sin(x) - x / 2)
)

# Root ≈ 0.6190612867
pr5 = Dict(
    :a => 0.0,
    :b => 1.0,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> exp(x) - 3x)
)

# Root ≈ 1.5596104694
pr6 = Dict(
    :a => 1.5,
    :b => 1.6,  # or wider [1,2]
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> log(x) + x - 2)
)

# Root = 1.0 (multiplicity 7, very flat)
pr7 = Dict(
    :a => 0.5,
    :b => 2.0,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> (x - 1)^7)
)

# Root = 1.0 (flat Gaussian-like)
pr8 = Dict(
    :a => 0.6,
    :b => 1.6,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> (x - 1) * exp(-100 * (x - 1)^2))
)

# Root ≈ 4.4934094579 (careful: tan has asymptote at π/2, 3π/2, etc.)
pr9 = Dict(
    :a => 4.3,
    :b => 4.6,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> tan(x) - x)
)

# Root ≈ -3.1831
pr10 = Dict(
    :a => -4,
    :b => -3,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> exp(x) - sin(x))
)

pr11 = Dict(
    :a => 2.7,
    :b => 3.3,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> x^2 - 5 * x + 6)
)

pr_midterm_5_3 = Dict(
    :a => -0.8,
    :b => -0.6,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> exp(x) - x^2)
)

# Define color schemes
const SUCCESS = Crayon(foreground=:green, bold=true)
const FAILURE = Crayon(foreground=:red, bold=true)
const INFO = Crayon(foreground=:cyan, bold=true)
const WARNING = Crayon(foreground=:yellow, bold=true)
const RESET = Crayon(reset=true)

function test_all_problems(verbose=false)
    problems = [
        ("pr1", pr1, "cos(x) - x"),
        ("pr2", pr2, "exp(-x) - x"),
        ("pr3", pr3, "x³ - x - 1"),
        ("pr4", pr4, "sin(x) - x/2"),
        ("pr5", pr5, "exp(x) - 3x"),
        ("pr6", pr6, "log(x) + x - 2"),
        ("pr7", pr7, "(x-1)⁷"),
        ("pr8", pr8, "(x-1)exp(-100(x-1)²)"),
        ("pr9", pr9, "tan(x) - x"),
        ("pr10", pr10, "exp(x) - sin(x)"),
        ("pr11", pr11, "x² - 5x + 6"),
        ("pr_midterm_5_3", pr_midterm_5_3, "exp(x) - x²")
    ]
    
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "Testing All Problems: Bisection vs Ridders", RESET)
    println(INFO, "=" ^ 80, RESET)
    
    results = []
    
    for (name, prob, desc) in problems
        println(INFO, "\n" * "-" ^ 80, RESET)
        println(INFO, "Problem: $name | f(x) = $desc", RESET)
        println(INFO, "Interval: [$(prob[:a]), $(prob[:b])], ϵ=$(prob[:ϵ]), δ=$(prob[:δ])", RESET)
        println(INFO, "-" ^ 80, RESET)
        
        # Test Bisection
        print("Bisection: ")
        bisection_result = nothing
        bisection_root = nothing
        try
            bisection_result = bisection_search(prob, verbose=verbose)
            bisection_root = bisection_result[1]
            if bisection_root !== nothing
                println(SUCCESS, "✓ Root = $bisection_root", RESET)
            else
                println(FAILURE, "✗ Failed to converge", RESET)
            end
        catch e
            println(FAILURE, "✗ Error: $(typeof(e))", RESET)
        end
        
        # Test Ridders
        print("Ridders:   ")
        ridders_result = nothing
        ridders_root = nothing
        try
            ridders_result = ridders_search(prob, verbose=verbose)
            ridders_root = ridders_result[1]
            if ridders_root !== nothing
                println(SUCCESS, "✓ Root = $ridders_root", RESET)
            else
                println(FAILURE, "✗ Failed to converge", RESET)
            end
        catch e
            println(FAILURE, "✗ Error: $(typeof(e))", RESET)
        end
        
        # Compare results
        if bisection_root !== nothing && ridders_root !== nothing
            abs_diff = abs(bisection_root - ridders_root)
            rel_diff = abs_diff / max(abs(bisection_root), abs(ridders_root), 1e-15)
            
            println("\nComparison:")
            println("  Absolute difference: ", WARNING, @sprintf("%.3e", abs_diff), RESET)
            println("  Relative difference: ", WARNING, @sprintf("%.3e", rel_diff), RESET)
            
            # Verify both are actually roots
            f_bisection = prob[:f](bisection_root)
            f_ridders = prob[:f](ridders_root)
            println("  |f(x_bisection)|: ", @sprintf("%.3e", abs(f_bisection)))
            println("  |f(x_ridders)|:   ", @sprintf("%.3e", abs(f_ridders)))
            
            push!(results, (name, desc, bisection_root, ridders_root, abs_diff, rel_diff, 
                        abs(f_bisection), abs(f_ridders), true, true))
        else
            push!(results, (name, desc, bisection_root, ridders_root, nothing, nothing,
                        nothing, nothing, bisection_root !== nothing, ridders_root !== nothing))
        end
    end
    
    # Summary table
    println(INFO, "\n\n" * "=" ^ 80, RESET)
    println(INFO, "SUMMARY TABLE", RESET)
    println(INFO, "=" ^ 80, RESET)
    println(@sprintf("%-20s %-12s %-12s %-12s %-12s", "Problem", "Bisection", "Ridders", "Abs Diff", "Rel Diff"))
    println("-" ^ 80)
    
    for result in results
        name, desc, b_root, r_root, abs_diff, rel_diff, f_b, f_r, b_success, r_success = result
        
        b_str = b_success ? (b_root !== nothing ? @sprintf("%.8f", b_root) : "Failed") : "Error"
        r_str = r_success ? (r_root !== nothing ? @sprintf("%.8f", r_root) : "Failed") : "Error"
        
        if abs_diff !== nothing
            diff_str = @sprintf("%.3e", abs_diff)
            rel_str = @sprintf("%.3e", rel_diff)
            
            status = (abs_diff < 1e-6) ? SUCCESS : WARNING
            print(status)
            println(@sprintf("%-20s %-12s %-12s %-12s %-12s", name, b_str, r_str, diff_str, rel_str))
            print(RESET)
        else
            print(FAILURE)
            println(@sprintf("%-20s %-12s %-12s %-12s %-12s", name, b_str, r_str, "N/A", "N/A"))
            print(RESET)
        end
    end
    
    println(INFO, "=" ^ 80, RESET)
    
    return results
end

# Add Printf for formatted output
using Printf

function ridders_search_with_tracking(pr; verbose=false)
    """
    Modified Ridders method that tracks convergence at each iteration.
    Returns: (root, f_root, iteration_history)
    where iteration_history contains: [itr, x0, x2, x1, x3, f0, f1, f2, f3, interval_size]
    """
    @unpack a, b, M, ϵ, δ, f = pr;
    itr = 1
    shouldStop = false
    
    history = []
    
    # Initial bracketing interval [x0, x2]
    x0 = a
    x2 = b
    f0 = f(x0)
    f2 = f(x2)
    
    sgn_f0 = sign(f0)
    sgn_f2 = sign(f2)
    
    if sgn_f0 * sgn_f2 > 0
        shouldStop = true
        error("f(x0) and f(x2) must have opposite signs for ridders_search to work.")
        return nothing, nothing, history
    end
    
    while !shouldStop
        # Compute midpoint x1
        x1 = (x0 + x2) / 2
        f1 = f(x1)
        
        # Compute distance d
        d = x2 - x0
        
        myprintln(verbose, "itr=$itr, x0=$x0, x2=$x2, x1=$x1, f0=$f0, f1=$f1, f2=$f2, d=$d")
        
        # Check convergence on interval size
        if abs(d) < δ
            shouldStop = true
            @error("Interval sufficiently small: |x2 - x0| = $d < δ = $δ")
            return nothing, nothing, history
        end
        
        # Compute parameter a for h(x) = f(x)*e^(a*x)
        discriminant = f1^2 - f2 * f0
        
        if discriminant < 0
            shouldStop = true
            @error("Negative discriminant: f1^2 - f2*f0 = $discriminant < 0")
            return nothing, nothing, history
        end
        
        a_param = (2 / d) * log((f1 + sign(f2) * sqrt(discriminant)) / f2)
        
        # Compute h(x) values
        h0 = f0 * exp(a_param * x0)
        h1 = f1 * exp(a_param * x1)
        h2 = f2 * exp(a_param * x2)
        
        myprintln(verbose, "    a_param=$a_param, h0=$h0, h1=$h1, h2=$h2")
        
        # Compute x3 (x-intercept of line through (x0, h0) and (x2, h2))
        if abs(h2 - h1) < 1e-14
            shouldStop = true
            @error("h2 - h1 too small: cannot compute x3")
            return nothing, nothing, history
        end
        
        x3 = (x1 * h2 - x2 * h1) / (h2 - h1)
        f3 = f(x3)
        
        # Store iteration data
        push!(history, (itr, x0, x2, x1, x3, f0, f1, f2, f3, d, abs(f3)))
        
        myprintln(verbose, "    x3=$x3, f3=$f3")
        
        # Check convergence on function value
        if abs(f3) < ϵ
            shouldStop = true
            println("Root found at x3 = $x3 with f(x3) = $f3")
            return x3, f3, history
        end
        
        # Determine next bracketing interval
        sgn_f1 = sign(f1)
        sgn_f3 = sign(f3)
        
        if sgn_f1 * sgn_f3 < 0
            # Root is in [x1, x3] or [x3, x1]
            if x1 < x3
                x0 = x1
                x2 = x3
                f0 = f1
                f2 = f3
            else
                x0 = x3
                x2 = x1
                f0 = f3
                f2 = f1
            end
        elseif sgn_f0 * sgn_f3 < 0
            # Root is in [x0, x3]
            x2 = x3
            f2 = f3
        elseif sgn_f2 * sgn_f3 < 0
            # Root is in [x3, x2]
            x0 = x3
            f0 = f3
        else
            shouldStop = true
            @error("Cannot determine next bracketing interval")
            return nothing, nothing, history
        end
        
        sgn_f0 = sign(f0)
        sgn_f2 = sign(f2)
        
        itr += 1
        if itr > M
            shouldStop = true
            @error("Exceeded maximum iterations M = $M")
            return nothing, nothing, history
        end
    end
end

function analyze_midterm_convergence()
    # Use tighter tolerance and wider interval to see more iterations
    pr_midterm = Dict(
        :a => -1.5,
        :b => 0.5,
        :M => 100,
        :ϵ => 1e-12,  # Much tighter tolerance
        :δ => 1e-12,
        :f => (x -> exp(x) - x^2)
    )
    
    println(INFO, "\n" * "=" ^ 80, RESET)
    println(INFO, "MIDTERM PROBLEM 5 - CONVERGENCE ANALYSIS", RESET)
    println(INFO, "Problem: f(x) = exp(x) - x²", RESET)
    println(INFO, "Interval: [$(pr_midterm[:a]), $(pr_midterm[:b])], ϵ=$(pr_midterm[:ϵ]), δ=$(pr_midterm[:δ])", RESET)
    println(INFO, "=" ^ 80, RESET)
    
    # Run Ridders with tracking
    println(INFO, "\nRunning Ridders Method with convergence tracking...\n", RESET)
    root, f_root, history = ridders_search_with_tracking(pr_midterm, verbose=true)
    
    if root !== nothing
        println(SUCCESS, "\n✓ Converged successfully!", RESET)
        println("Final root: x = ", SUCCESS, @sprintf("%.15f", root), RESET)
        println("Final f(x): ", SUCCESS, @sprintf("%.15e", f_root), RESET)
        println("Number of iterations: ", SUCCESS, length(history), RESET)
        
        # Print convergence table
        println(INFO, "\n" * "=" ^ 80, RESET)
        println(INFO, "CONVERGENCE TABLE", RESET)
        println(INFO, "=" ^ 80, RESET)
        println(@sprintf("%-4s %-12s %-12s %-12s %-12s %-12s %-12s", 
                        "Itr", "x3", "|f(x3)|", "Interval", "x0", "x2", "d"))
        println("-" ^ 80)
        
        for (itr, x0, x2, x1, x3, f0, f1, f2, f3, d, abs_f3) in history
            println(@sprintf("%-4d %-12.8f %-12.3e %-12.3e %-12.8f %-12.8f %-12.3e", 
                            itr, x3, abs_f3, d, x0, x2, d))
        end
        
        # Convergence rate analysis
        if length(history) > 1
            println(INFO, "\n" * "=" ^ 80, RESET)
            println(INFO, "CONVERGENCE RATE ANALYSIS", RESET)
            println(INFO, "=" ^ 80, RESET)
            println(@sprintf("%-4s %-15s %-15s %-15s", "Itr", "|f(x3)|", "Ratio", "Order"))
            println("-" ^ 80)
            
            for i in 1:length(history)
                itr, _, _, _, _, _, _, _, _, _, abs_f3 = history[i]
                if i > 1
                    prev_abs_f3 = history[i-1][11]
                    ratio = abs_f3 / prev_abs_f3
                    order = -log(ratio)
                    println(@sprintf("%-4d %-15.3e %-15.3e %-15.3f", itr, abs_f3, ratio, order))
                else
                    println(@sprintf("%-4d %-15.3e %-15s %-15s", itr, abs_f3, "N/A", "N/A"))
                end
            end
            
            # Average convergence order
            if length(history) > 2
                orders = []
                for i in 2:length(history)
                    abs_f3_curr = history[i][11]
                    abs_f3_prev = history[i-1][11]
                    if abs_f3_prev > 0 && abs_f3_curr > 0
                        ratio = abs_f3_curr / abs_f3_prev
                        push!(orders, -log(ratio))
                    end
                end
                if !isempty(orders)
                    avg_order = sum(orders) / length(orders)
                    println(INFO, "\nAverage convergence order: ", WARNING, 
                           @sprintf("%.3f", avg_order), RESET)
                    println(INFO, "Theoretical Ridders order: ", WARNING, "1.839", RESET)
                end
            end
        end
        
        println(INFO, "\n" * "=" ^ 80, RESET)
    else
        println(FAILURE, "\n✗ Failed to converge", RESET)
    end
    
    return root, f_root, history
end

# Run the analysis
# test_all_problems(false)

# First show the original problem
println(INFO, "\n" * "=" ^ 80, RESET)
println(INFO, "PART 1: Original Problem Settings (as given in midterm)", RESET)
println(INFO, "=" ^ 80, RESET)

pr_original = Dict(
    :a => -0.8,
    :b => -0.6,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> exp(x) - x^2)
)

println(INFO, "Interval: [-0.8, -0.6], ϵ=1e-5, δ=1e-5\n", RESET)
root_orig, f_orig, history_orig = ridders_search_with_tracking(pr_original, verbose=false)
if root_orig !== nothing
    println(SUCCESS, "✓ Converged in $(length(history_orig)) iteration(s)", RESET)
    println("  Root: x = ", @sprintf("%.15f", root_orig))
    println("  f(x): ", @sprintf("%.3e", f_orig))
end

# Then show extended analysis
println(INFO, "\n" * "=" ^ 80, RESET)
println(INFO, "PART 2: Extended Analysis (wider interval, tighter tolerance)", RESET)
println(INFO, "=" ^ 80, RESET)

analyze_midterm_convergence()
