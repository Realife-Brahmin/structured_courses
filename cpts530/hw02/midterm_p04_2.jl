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

# Fixed point iteration function
function fixed_point_iteration(T; x0, tol=1e-10, maxiter=100, verbose=false)
    x_prev = x0
    for k in 1:maxiter
        x_next = T(x_prev)
        myprintln(verbose, "Iter $k: x_prev=$(x_prev), x_next=$(x_next), |x_next-x_prev|=$(abs(x_next-x_prev))")
        if abs(x_next - x_prev) < tol
            println(SUCCESS, "✓ Converged in $k iteration(s)", RESET)
            println("  Solution: x = ", @sprintf("%.10f", x_next))
            return x_next, k
        end
        x_prev = x_next
    end
    println(FAILURE, "✗ Did not converge in $maxiter iterations", RESET)
    return nothing, maxiter
end

# Problem function and T(x)
f(x) = x - 0.5 * (cos(x/2) - abs(x - 0.5))
T(x) = 0.5 * (cos(x/2) - abs(x - 0.5))

# Initial guess selection
x0 = 0.48  # Chosen within [0.45, 0.55] as shown in part 1

println(INFO, "\n" * "="^80, RESET)
println(INFO, "MIDTERM PROBLEM 4.2: Fixed Point Iteration", RESET)
println(INFO, "="^80, RESET)
println("Initial guess: x0 = ", x0)

# Run fixed point iteration
tol = 1e-10
maxiter = 100
solution, n_iter = fixed_point_iteration(T; x0=x0, tol=tol, maxiter=maxiter, verbose=true)

if solution !== nothing
    println(SUCCESS, "\nFinal solution: x = ", @sprintf("%.10f", solution), RESET)
    println("Number of iterations: ", n_iter)
    println("Check: f(x) = ", @sprintf("%.3e", f(solution)))
else
    println(FAILURE, "No solution found.", RESET)
end
