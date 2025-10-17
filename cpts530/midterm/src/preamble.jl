# Preamble: Environment setup and imports
# This file contains all the setup code needed at the start

# Activate the cpts530 environment
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using LinearAlgebra
using Printf
using Crayons

# Define color schemes for output
const SUCCESS = Crayon(foreground=:green, bold=true)
const FAILURE = Crayon(foreground=:red, bold=true)
const INFO = Crayon(foreground=:cyan, bold=true)
const WARNING = Crayon(foreground=:yellow, bold=true)
const RESET = Crayon(reset=true)
