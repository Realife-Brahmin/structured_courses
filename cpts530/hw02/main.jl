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
    :a => 0.0,
    :b => 2.0,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> x^3 - 6x^2 + 11x - 6)
)
bisection_search(pr, verbose=true)
