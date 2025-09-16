using Parameters

function bisection_search(pr)
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

pr = Dict(
    :a => 2.7,
    :b => 3.3,
    :M => 100,
    :ϵ => 1e-5,
    :δ => 1e-5,
    :f => (x -> x^2 - 5*x + 6)
)


bisection_search(pr)
