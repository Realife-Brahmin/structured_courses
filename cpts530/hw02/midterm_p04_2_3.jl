# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Parameters
using Crayons
using Printf

# Print helpers
myprintln(args...) = println(args...)
myprintln(verbose::Bool, args...) = verbose && println(args...)
myprintln(io::IO, args...) = println(io, args...)
myprintln(io::IO, verbose::Bool, args...) = verbose && println(io, args...)

# Color schemes
const SUCCESS = Crayon(foreground=:green, bold=true)
const FAILURE = Crayon(foreground=:red, bold=true)
const INFO = Crayon(foreground=:cyan, bold=true)
const WARNING = Crayon(foreground=:yellow, bold=true)
const RESET = Crayon(reset=true)

# Problem function and T(x)
f(x) = x - 0.5 * (cos(x/2) - abs(x - 0.5))
T(x) = 0.5 * (cos(x/2) - abs(x - 0.5))

# Contraction constant (lambda)
lambda = 0.5679  # Use the correct upper bound for the interval

# Initial guess and tolerance
x0 = 0.48
x1 = T(x0)
interval_gap = abs(x1 - x0)
tol = 1e-10
maxiter = 100

println(INFO, "\n" * "="^80, RESET)
println(INFO, "MIDTERM PROBLEM 4.2/4.3: Fixed Point Iteration & Error Estimates", RESET)
println(INFO, "="^80, RESET)
println("Initial guess: x0 = ", x0)
println("Contraction constant (lambda): ", @sprintf("%.4f", lambda))

# A priori estimate for required iterations
function apriori_iterations(lambda, gap, tol)
    n = log(tol * (1 - lambda) / gap) / log(lambda)
    return ceil(Int, n)
end

n_apriori = apriori_iterations(lambda, interval_gap, tol)
println(WARNING, "A priori estimate: at least $n_apriori iterations needed for accuracy $tol", RESET)

# Fixed point iteration with a posteriori error tracking
function fixed_point_iteration_with_error(T; x0, tol=1e-10, maxiter=100, lambda=0.5679, verbose=false)
    x_prev = x0
    errors = []
    for k in 1:maxiter
        x_next = T(x_prev)
        err_aposteriori = lambda / (1 - lambda) * abs(x_next - x_prev)
        push!(errors, err_aposteriori)
        myprintln(verbose, "Iter $k: x_prev=$(x_prev), x_next=$(x_next), |x_next-x_prev|=$(abs(x_next-x_prev)), a posteriori error=$(err_aposteriori)")
        if abs(x_next - x_prev) < tol
            println(SUCCESS, "✓ Converged in $k iteration(s)", RESET)
            println("  Solution: x = ", @sprintf("%.10f", x_next))
            println("  Last a posteriori error estimate: ", @sprintf("%.3e", err_aposteriori))
            return x_next, k, errors
        end
        x_prev = x_next
    end
    println(FAILURE, "✗ Did not converge in $maxiter iterations", RESET)
    return nothing, maxiter, errors
end

solution, n_iter, errors = fixed_point_iteration_with_error(T; x0=x0, tol=tol, maxiter=maxiter, lambda=lambda, verbose=true)

if solution !== nothing
    println(SUCCESS, "\nFinal solution: x = ", @sprintf("%.10f", solution), RESET)
    println("Number of iterations: ", n_iter)
    println("Check: f(x) = ", @sprintf("%.3e", f(solution)))
else
    println(FAILURE, "No solution found.", RESET)
end

println(INFO, "\nA priori estimate: $n_apriori iterations required.", RESET)
println(INFO, "Actual iterations: $n_iter.", RESET)
println(INFO, "A posteriori error estimates per iteration:", RESET)
for (k, err) in enumerate(errors)
    println(@sprintf("Iter %2d: a posteriori error = %.3e", k, err))
end
