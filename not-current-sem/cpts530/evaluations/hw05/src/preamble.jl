# Preamble: Environment setup and imports
# CPTS530 HW05

# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using LinearAlgebra
using Printf
using Plots
using Crayons

# Define color schemes for output
const SUCCESS = Crayon(foreground=:green, bold=true)
const FAILURE = Crayon(foreground=:red, bold=true)
const INFO = Crayon(foreground=:cyan, bold=true)
const WARNING = Crayon(foreground=:yellow, bold=true)
const RESET = Crayon(reset=true)

# Set default plot settings
default(
    fontfamily="Computer Modern",
    linewidth=2,
    framestyle=:box,
    grid=true,
    legend=:best,
    size=(800, 600)
)

println(INFO, "✓ Environment activated: cpts530", RESET)
println(INFO, "✓ Packages loaded successfully", RESET)
