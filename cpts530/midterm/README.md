# Midterm Problem 3: Code Organization

## File Structure

```
cpts530/midterm/
├── midterm_p03.jl          # Main file - Core LU functions (END-USER EDITS THIS)
├── src/
│   ├── preamble.jl         # Environment setup, imports, color constants
│   └── test_utils.jl       # Testing functions and test problems
└── README.md               # This file
```

## Description

### `midterm_p03.jl` (Main File - ~120 lines)
**This is the file you should edit as an end-user.**

Contains the core LU factorization functions:
- `get_LU_factors(A)` - ✅ Implemented: Doolittle LU decomposition
- `solve_Ly_is_equal_to_b(L, b)` - ⏳ TODO: Forward substitution
- `solve_Ux_is_equal_to_y(U, y)` - ⏳ TODO: Backward substitution
- `solve_linear_system_LU(A, b)` - Driver function combining all steps

### `src/preamble.jl` (~18 lines)
**Do not edit unless changing environment setup.**

Contains:
- Package activation: `Pkg.activate()`
- Imports: `LinearAlgebra`, `Printf`, `Crayons`
- Color constants: `SUCCESS`, `FAILURE`, `INFO`, `WARNING`, `RESET`

### `src/test_utils.jl` (~260 lines)
**Do not edit unless changing test framework.**

Contains:
- Verification functions:
  - `verify_LU_factorization(A, L, U)` - Checks A = LU
  - `verify_solution(A, x, b)` - Checks Ax = b
  
- Testing functions:
  - `test_LU_only(A)` - Quick test for LU factorization
  - `test_LU_solver(A, b)` - Full system solving test with comparison to built-in
  
- Test problems:
  - `test_problem_1()` - 3×3 midterm system
  - `test_problem_2()` - 2×2 simple system
  - `test_problem_identity()` - Identity matrix
  - `test_problem_diagonal()` - Diagonal matrix

## Usage

### Running Tests
```bash
cd cpts530/midterm
julia midterm_p03.jl
```

This will run 4 LU factorization tests (all currently pass with 0 error).

### Implementing Forward/Backward Substitution

Edit `midterm_p03.jl` and fill in:
```julia
function solve_Ly_is_equal_to_b(L, b)
    n = length(b)
    y = zeros(n)
    
    # YOUR IMPLEMENTATION HERE
    for i = 1:n
        y[i] = (b[i] - sum(L[i,j]*y[j] for j=1:i-1)) / L[i,i]
    end
    
    return y
end

function solve_Ux_is_equal_to_y(U, y)
    n = length(y)
    x = zeros(n)
    
    # YOUR IMPLEMENTATION HERE
    for i = n:-1:1
        x[i] = (y[i] - sum(U[i,j]*x[j] for j=i+1:n)) / U[i,i]
    end
    
    return x
end
```

### Testing Full System Solving

After implementing forward/backward substitution, uncomment test problems in the main execution block:
```julia
# test_problem_1()      # 3×3 system
# test_problem_2()      # 2×2 system
# test_problem_identity()  # Identity
# test_problem_diagonal()  # Diagonal
```

## Current Status

✅ **Completed:**
- LU factorization (`get_LU_factors`) - Passes all 4 tests with max error = 0.000e+00
- Refactored code into modular structure
- Comprehensive testing framework

⏳ **TODO:**
- Implement `solve_Ly_is_equal_to_b()` (forward substitution)
- Implement `solve_Ux_is_equal_to_y()` (backward substitution)
- Run full system solving tests

## Test Results (Current)

All LU factorization tests pass:
- Test 1 (2×2): ✅ max error = 0.000e+00
- Test 2 (3×3 midterm): ✅ max error = 0.000e+00
- Test 3 (Identity): ✅ max error = 0.000e+00
- Test 4 (Diagonal): ✅ max error = 0.000e+00
