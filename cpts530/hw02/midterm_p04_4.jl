# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Parameters
using Crayons
using Printf

# Print helpers
myprintln(args...) = println(args...)
myprintln(verbose::Bool, args...) = verbose && println(args...)

const SUCCESS = Crayon(foreground=:green, bold=true)
const FAILURE = Crayon(foreground=:red, bold=true)
const INFO = Crayon(foreground=:cyan, bold=true)
const WARNING = Crayon(foreground=:yellow, bold=true)
const RESET = Crayon(reset=true)

# Problem function and its derivative
f(x) = x - 0.5 * (cos(x/2) - abs(x - 0.5))
function fp(x)
    # Derivative: 1 - 0.25*sin(x/2) - 0.5*sgn(x-0.5), except at x=0.5
    if isapprox(x, 0.5; atol=1e-10)
        return nothing  # Not defined at x=0.5
    else
        return 1 + 0.25 * sin(x/2) + 0.5 * sign(x - 0.5)
    end
end

# FPI fallback
T(x) = 0.5 * (cos(x/2) - abs(x - 0.5))


# True root from previous FPI: x* ≈ 0.4722515915
const xstar = 0.4722515915


function newton_with_fallback(f, fp, T; x0, tol=1e-10, maxiter=100, verbose=false)
    x = x0
    errors = []
    xs = [x0]
    for k in 1:maxiter
        fval = f(x)
        fpval = fp(x)
        err = abs(x - xstar)
        push!(errors, err)
        myprintln(verbose, "Iter $k: x=$(x), f(x)=$(fval), f'(x)=$(fpval), error=$(err)")
        if abs(fval) < tol
            println(SUCCESS, "✓ Newton converged in $k iteration(s)", RESET)
            println("  Solution: x = ", @sprintf("%.10f", x))
            return x, k, :newton, errors, xs
        end
        if fpval === nothing || abs(fpval) < 1e-12
            println(WARNING, "Switching to fixed point iteration at x = ", x, RESET)
            # FPI fallback
            x_fpi = x
            for j in 1:maxiter
                x_next = T(x_fpi)
                err_fpi = abs(x_next - xstar)
                myprintln(verbose, "  FPI Iter $j: x=$(x_fpi), x_next=$(x_next), error=$(err_fpi)")
                if abs(x_next - x_fpi) < tol
                    println(SUCCESS, "✓ FPI converged in $j iteration(s)", RESET)
                    println("  Solution: x = ", @sprintf("%.10f", x_next))
                    return x_next, k+j, :fpi, errors, xs
                end
                x_fpi = x_next
                push!(errors, err_fpi)
                push!(xs, x_next)
            end
            println(FAILURE, "✗ FPI did not converge in fallback", RESET)
            return nothing, k+maxiter, :fpi, errors, xs
        end
        x_new = x - fval / fpval
        push!(xs, x_new)
        if abs(x_new - x) < tol
            println(SUCCESS, "✓ Newton converged in $k iteration(s)", RESET)
            println("  Solution: x = ", @sprintf("%.10f", x_new))
            return x_new, k, :newton, errors, xs
        end
        x = x_new
    end
    println(FAILURE, "✗ Newton did not converge in $maxiter iterations", RESET)
    return nothing, maxiter, :newton, errors, xs
end

# Initial guess and tolerance
x0 = 0.48

println(INFO, "\n" * "="^80, RESET)
println(INFO, "MIDTERM PROBLEM 4.4: Newton's Method with FPI Fallback", RESET)
println(INFO, "="^80, RESET)
println("Initial guess: x0 = ", x0)

solution, n_iter, method, errors, xs = newton_with_fallback(f, fp, T; x0=x0, tol=1e-10, maxiter=100, verbose=true)

if solution !== nothing
    println(SUCCESS, "\nFinal solution: x = ", @sprintf("%.10f", solution), RESET)
    println("Number of iterations: ", n_iter)
    println("Method used: ", method)
    println("Check: f(x) = ", @sprintf("%.3e", f(solution)))
    println(INFO, "\nConvergence Table:", RESET)
    println(@sprintf("%-4s %-16s %-12s %-16s", "n", "x_n", "e_n", "e_{n+1}/e_{n}^2"))
    println("-"^60)
    for k in 1:length(errors)-1
        e_n = errors[k]
        e_np1 = errors[k+1]
        ratio = e_n == 0 ? "-" : @sprintf("%.3e", e_np1 / e_n^2)
        println(@sprintf("%-4d %-16.10f %-12.3e %-16s", k-1, xs[k], e_n, ratio))
    end
    # Print last row
    k = length(errors)
    println(@sprintf("%-4d %-16.10f %-12.3e", k-1, xs[k], errors[k]))
else
    println(FAILURE, "No solution found.", RESET)
end
