using Parameters

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

println("=" ^ 60)
println("Testing Bisection Search on pr_midterm_5_3")
println("=" ^ 60)
bisection_search(pr_midterm_5_3, verbose=true)

println("\n" * "=" ^ 60)
println("Testing Ridders Search on pr_midterm_5_3")
println("=" ^ 60)
ridders_search(pr_midterm_5_3, verbose=true)
