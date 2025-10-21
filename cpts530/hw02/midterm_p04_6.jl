# =====================================================================
# MIDTERM PROBLEM 4.6: Fixed Point Iteration (FPI) Convergence Analysis
# =====================================================================
# This script performs FPI for the nonlinear equation:
#     x = 0.5 * (cos(x/2) - |x - 0.5|)
# starting from x0 = 0.48, with stopping criterion |x_{n+1} - x_n| < 1e-10.
# It prints a colored convergence table with n, x_n, approximate error, and error ratio.
# =====================================================================

# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Crayons
using Printf

const SUCCESS = Crayon(foreground=:green, bold=true)
const FAILURE = Crayon(foreground=:red, bold=true)
const INFO = Crayon(foreground=:cyan, bold=true)
const WARNING = Crayon(foreground=:yellow, bold=true)
const HEADER = Crayon(foreground=:magenta, bold=true)
const VALUE = Crayon(foreground=:white)
const RATIO_GOOD = Crayon(foreground=:blue, bold=true)
const RATIO_BAD = Crayon(foreground=:red, bold=true)
const RESET = Crayon(reset=true)

# Fixed Point Iteration function T(x)
T(x) = 0.5 * (cos(x/2) - abs(x - 0.5))

# Parameters
x0 = 0.48
max_iter = 25
const tol = 1e-10

# Store iterates
xs = [x0]

println(INFO, "\n" * "="^80, RESET)
println(INFO, "MIDTERM PROBLEM 4.6: Fixed Point Iteration (FPI) Convergence Analysis", RESET)
println(INFO, "="^80, RESET)
println(INFO, "Equation: x = 0.5 * (cos(x/2) - |x - 0.5|)", RESET)
println(INFO, "Initial guess: x0 = $x0", RESET)
println(INFO, "Tolerance: tol = $tol", RESET)
println(INFO, "Stopping criterion: |x_{n+1} - x_n| < tol", RESET)

println(INFO, "Convergence Table:", RESET)
println(HEADER, "n    x_n           e_n (approx)           e_{n+1}/e_n", RESET)
println(HEADER, "------------------------------------------------------", RESET)

for n in 1:max_iter
    x_next = T(xs[end])
    push!(xs, x_next)
    # Approximate error: e_n ≈ |x_n - x_{n-1}|
    e_n = abs(xs[end] - xs[end-1])
    if n > 1
        e_prev = abs(xs[end-1] - xs[end-2])
        ratio = e_n / e_prev
        ratio_color = ratio < 1 ? RATIO_GOOD : RATIO_BAD
        @printf("%s%2d%s %s%.10f%s %s%16.3e%s %s%16.3e%s\n", HEADER, n, RESET, VALUE, x_next, RESET, VALUE, e_n, RESET, ratio_color, ratio, RESET)
    else
        @printf("%s%2d%s %s%.10f%s %s%16.3e%s      --\n", HEADER, n, RESET, VALUE, x_next, RESET, VALUE, e_n, RESET)
    end
    if e_n < tol
        println(SUCCESS, "\n✓ FPI converged: |x_{n+1} - x_n| < tol = ", tol, RESET)
        println(INFO, "Final solution: x = ", @sprintf("%.10f", x_next), RESET)
        println(INFO, "Iterations: ", n, RESET)
        break
    end
end

# True root from previous FPI
const r = 0.4722515915

println(INFO, "\nA posteriori vs. true error table:", RESET)
println(HEADER, "n    x_n           |x_n - r|         a posteriori bound", RESET)
println(HEADER, "------------------------------------------------------", RESET)

# Contraction constant (estimate from T'(r))
lambda = abs(-0.25 * sin(r/2) - 0.5 * sign(r - 0.5))

for n in 2:length(xs)
    x_n = xs[n]
    true_err = abs(x_n - r)
    apos_err = lambda / (1 - lambda) * abs(x_n - xs[n-1])
    @printf("%s%2d%s %s%.10f%s %s%16.3e%s %s%16.3e%s\n", HEADER, n-1, RESET, VALUE, x_n, RESET, VALUE, true_err, RESET, VALUE, apos_err, RESET)
end