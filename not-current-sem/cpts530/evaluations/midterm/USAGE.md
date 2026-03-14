# How to Use midterm_p03.jl - Usage Examples

## Overview

The `midterm_p03.jl` file has **two boolean flags** at the top that control how it runs:

```julia
const RUN_TESTS = true          # Enable/disable automatic testing
const TEST_FULL_SOLVER = false  # Test LU only or full system solving
```

---

## Mode 1: Automatic LU Factorization Testing (DEFAULT)

**Settings:**
```julia
const RUN_TESTS = true
const TEST_FULL_SOLVER = false
```

**What it does:**
- Runs 4 comprehensive LU factorization tests automatically
- Tests: 2×2, 3×3 midterm, identity, and diagonal matrices
- Shows detailed output with L, U matrices and error verification

**Run it:**
```bash
cd cpts530/midterm
julia midterm_p03.jl
```

**Output:**
```
================================================================================
MIDTERM PROBLEM 3: LU FACTORIZATION TESTS
================================================================================

TEST 1: Simple 2×2 Matrix
...
✓ LU factorization CORRECT! (max error < 1e-10)

TEST 2: Midterm Problem 3×3 Matrix
...
✓ LU factorization CORRECT! (max error < 1e-10)
...
```

---

## Mode 2: Full System Solving Tests

**Settings:**
```julia
const RUN_TESTS = true
const TEST_FULL_SOLVER = true  # CHANGED!
```

**What it does:**
- Runs full Ax = b system solving tests
- Uses forward/backward substitution
- Compares your solution against Julia's built-in solver
- **Use this AFTER implementing `solve_Ly_is_equal_to_b()` and `solve_Ux_is_equal_to_y()`**

**Run it:**
```bash
cd cpts530/midterm
julia midterm_p03.jl
```

**Output:**
```
================================================================================
FULL SYSTEM SOLVING TESTS
================================================================================

Testing: Problem 1: Simple 3×3 system
...
✓ LU factorization verified: max error = 0.000e+00
✓ Solution verified: max residual = 0.000e+00
✓ Solutions match!
...
```

---

## Mode 3: Manual/Interactive Mode

**Settings:**
```julia
const RUN_TESTS = false  # CHANGED!
const TEST_FULL_SOLVER = false
```

**What it does:**
- Loads all functions but runs NO automatic tests
- Perfect for:
  - Testing specific matrices manually
  - Debugging individual functions
  - Interactive REPL usage
  - Writing your own test code

**Run it:**
```bash
cd cpts530/midterm
julia midterm_p03.jl
```

**Output:**
```
================================================================================
MANUAL MODE: Tests disabled (RUN_TESTS = false)
================================================================================

You can now use functions interactively:
  - get_LU_factors(A)
  - solve_Ly_is_equal_to_b(L, b)
  - solve_Ux_is_equal_to_y(U, y)
  - solve_linear_system_LU(A, b)

Or call specific tests:
  - test_LU_only(A, name="My Test")
  - test_problem_1(), test_problem_2(), etc.
================================================================================
```

**Then in the file, add your own test code at the bottom:**
```julia
    else
        # Manual mode - add your test here!
        println(INFO, "\nMy custom test:", RESET)
        
        # Test just one matrix
        A = [2.0 1.0 1.0; 4.0 -6.0 0.0; -2.0 7.0 2.0]
        L, U = get_LU_factors(A)
        
        println("\nA = "); display(A)
        println("\nL = "); display(L)
        println("\nU = "); display(U)
        
        # Or call a specific test function
        test_LU_only(A, name="My custom test")
    end
```

---

## Mode 4: Include in Another Script

You can also `include()` this file in another Julia script to use the functions:

**my_experiments.jl:**
```julia
# Load the LU functions (but don't run the main block)
include("midterm_p03.jl")

# Now use the functions
A = [4.0 3.0; 6.0 3.0]
L, U = get_LU_factors(A)

println("My own test:")
display(L)
display(U)

# Or use the test utilities
test_LU_only(A, name="My experiment", show_matrices=false)
```

**Run it:**
```bash
julia my_experiments.jl
```

---

## Quick Reference Table

| Mode | RUN_TESTS | TEST_FULL_SOLVER | Use Case |
|------|-----------|------------------|----------|
| **LU Tests Only** | `true` | `false` | Default: Test LU factorization |
| **Full System Tests** | `true` | `true` | After implementing forward/backward substitution |
| **Manual/Custom** | `false` | `false` | Write your own tests, debug specific cases |
| **Include Mode** | N/A | N/A | Use functions in another script |

---

## Typical Workflow

### Step 1: Verify LU Factorization Works
```julia
const RUN_TESTS = true
const TEST_FULL_SOLVER = false
```
```bash
julia midterm_p03.jl
# Should see: ✓ LU factorization CORRECT! (4/4 tests pass)
```

### Step 2: Implement Forward/Backward Substitution
Edit the TODO functions in `midterm_p03.jl`:
- `solve_Ly_is_equal_to_b(L, b)`
- `solve_Ux_is_equal_to_y(U, y)`

### Step 3: Test Full System Solving
```julia
const RUN_TESTS = true
const TEST_FULL_SOLVER = true  # Now ready!
```
```bash
julia midterm_p03.jl
# Should see: ✓ Solutions match! (comparing with Julia's built-in)
```

### Step 4: Debug if Needed
```julia
const RUN_TESTS = false  # Manual mode
```
Add your own debug code at the bottom to test specific cases.

---

## Example: Testing Just One Matrix

**Set:**
```julia
const RUN_TESTS = false
```

**In the `else` block, add:**
```julia
        # Example: Test just the midterm matrix
        A = [2.0 1.0 1.0; 4.0 -6.0 0.0; -2.0 7.0 2.0]
        b = [5.0, -2.0, 9.0]
        
        println(INFO, "\nTesting midterm problem:", RESET)
        L, U = get_LU_factors(A)
        println("L = "); display(L)
        println("U = "); display(U)
        
        # Once forward/backward substitution is done:
        # x, L, U = solve_linear_system_LU(A, b)
        # println("x = "); display(x)
```

---

## Summary

✅ **RUN_TESTS = true, TEST_FULL_SOLVER = false** → Default mode, test LU factorization  
✅ **RUN_TESTS = true, TEST_FULL_SOLVER = true** → Full system solving tests  
✅ **RUN_TESTS = false** → Manual mode, write your own tests  
✅ **include("midterm_p03.jl")** → Use in another script  

The file is flexible and adapts to your development stage! 🎯
