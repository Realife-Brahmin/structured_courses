# 🎛️ QUICK START GUIDE: midterm_p03.jl

## 🚀 Three Simple Steps to Control Your Tests

### Step 1: Open `midterm_p03.jl`
### Step 2: Find these two lines (around line 120):
```julia
const RUN_TESTS = true          # <-- Change this!
const TEST_FULL_SOLVER = false  # <-- Or this!
```
### Step 3: Choose your mode and run!

---

## 🎯 Four Usage Modes

```
┌─────────────────────────────────────────────────────────────────┐
│  MODE 1: LU Factorization Testing (DEFAULT)                     │
├─────────────────────────────────────────────────────────────────┤
│  const RUN_TESTS = true                                         │
│  const TEST_FULL_SOLVER = false                                 │
│                                                                  │
│  $ julia midterm_p03.jl                                         │
│                                                                  │
│  ✓ Runs 4 LU factorization tests automatically                 │
│  ✓ Shows L, U matrices                                          │
│  ✓ Verifies A = LU                                              │
│  ✓ Shows max error (should be 0.000e+00)                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MODE 2: Full System Solving Tests                              │
├─────────────────────────────────────────────────────────────────┤
│  const RUN_TESTS = true                                         │
│  const TEST_FULL_SOLVER = true         👈 CHANGE THIS!          │
│                                                                  │
│  $ julia midterm_p03.jl                                         │
│                                                                  │
│  ✓ Tests complete Ax = b solving                                │
│  ✓ Uses forward/backward substitution                           │
│  ✓ Compares with Julia's built-in solver                        │
│  ✓ Use AFTER implementing solve_Ly and solve_Ux functions       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MODE 3: Manual/Interactive Mode                                │
├─────────────────────────────────────────────────────────────────┤
│  const RUN_TESTS = false               👈 CHANGE THIS!          │
│  const TEST_FULL_SOLVER = false                                 │
│                                                                  │
│  $ julia midterm_p03.jl                                         │
│                                                                  │
│  ✓ Loads functions but runs NO automatic tests                  │
│  ✓ Add your own test code in the else block                     │
│  ✓ Perfect for debugging specific matrices                      │
│  ✓ Interactive experimentation                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MODE 4: Include in Another Script                              │
├─────────────────────────────────────────────────────────────────┤
│  # In your_script.jl:                                           │
│  include("midterm_p03.jl")                                      │
│                                                                  │
│  A = [4.0 3.0; 6.0 3.0]                                         │
│  L, U = get_LU_factors(A)                                       │
│                                                                  │
│  ✓ Use functions in your own scripts                            │
│  ✓ Main execution block won't interfere                         │
│  ✓ Access all functions and test utilities                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 What You'll See in Each Mode

### Mode 1 Output (LU Tests):
```
================================================================================
MIDTERM PROBLEM 3: LU FACTORIZATION TESTS
================================================================================

TEST 1: Simple 2×2 Matrix
================================================================================
Testing LU Factorization: 2×2 system
Matrix size: 2 × 2
================================================================================

Original matrix A:
2×2 Matrix{Float64}:
 4.0  3.0
 6.0  3.0

L (Lower triangular):
2×2 Matrix{Float64}:
 1.0  0.0
 1.5  1.0

U (Upper triangular):
2×2 Matrix{Float64}:
 4.0   3.0
 0.0  -1.5

Max absolute error: 0.000e+00
✓ LU factorization CORRECT! (max error < 1e-10)
```

### Mode 2 Output (Full Solver):
```
================================================================================
FULL SYSTEM SOLVING TESTS
================================================================================

Testing: Problem 1: Simple 3×3 system
Matrix size: 3 × 3

Solving Ax = b using custom LU factorization...

Verifying LU factorization:
✓ LU factorization verified: max error = 0.000e+00

Verifying solution:
✓ Solution verified: max residual = 0.000e+00

Comparing with Julia's built-in solver:
Max difference from built-in: 0.000e+00
✓ Solutions match!

Solution comparison:
i    x_custom        x_builtin       difference
------------------------------------------------------------
1    1.000000e+00    1.000000e+00    0.000000e+00
2    2.000000e+00    2.000000e+00    0.000000e+00
3    3.000000e+00    3.000000e+00    0.000000e+00
```

### Mode 3 Output (Manual):
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

[Your custom code runs here]
```

---

## 🔧 Typical Development Workflow

```
1️⃣  Verify LU works
    ├─ Set: RUN_TESTS = true, TEST_FULL_SOLVER = false
    ├─ Run: julia midterm_p03.jl
    └─ See: ✓ LU factorization CORRECT! (4/4 tests)

2️⃣  Implement forward/backward substitution
    ├─ Edit: solve_Ly_is_equal_to_b(L, b)
    └─ Edit: solve_Ux_is_equal_to_y(U, y)

3️⃣  Test full system solving
    ├─ Set: RUN_TESTS = true, TEST_FULL_SOLVER = true
    ├─ Run: julia midterm_p03.jl
    └─ See: ✓ Solutions match!

4️⃣  Debug if needed
    ├─ Set: RUN_TESTS = false
    ├─ Add custom test code
    └─ Run specific cases
```

---

## 🎓 Example: Testing Just One Matrix

**In `midterm_p03.jl`, set:**
```julia
const RUN_TESTS = false
```

**In the `else` block (around line 220), add:**
```julia
        # Example: Test the midterm matrix only
        println(INFO, "\n🔍 Testing midterm matrix:", RESET)
        
        A = [2.0 1.0 1.0; 4.0 -6.0 0.0; -2.0 7.0 2.0]
        L, U = get_LU_factors(A)
        
        println("\nA ="); display(A)
        println("\nL ="); display(L)
        println("\nU ="); display(U)
        println("\nL*U ="); display(L*U)
        println("\nError ="); display(A - L*U)
```

**Run it:**
```bash
julia midterm_p03.jl
```

---

## 📚 Files Structure

```
midterm/
├── midterm_p03.jl          ← YOU EDIT THIS (core LU functions)
├── src/
│   ├── preamble.jl         ← Don't touch (imports & setup)
│   └── test_utils.jl       ← Don't touch (testing framework)
├── README.md               ← File structure overview
├── USAGE.md                ← Detailed usage guide (this file)
└── QUICKSTART.md           ← Quick reference (current file)
```

---

## 🎯 Quick Reference

| What I Want | RUN_TESTS | TEST_FULL_SOLVER |
|------------|-----------|------------------|
| Test LU factorization | `true` | `false` |
| Test full Ax=b solving | `true` | `true` |
| Write my own tests | `false` | any |
| Use in another script | N/A (use `include()`) | |

---

## ✨ Summary

**One file, four modes, controlled by TWO boolean flags!**

- ✅ `RUN_TESTS = true` → Automatic testing
- ✅ `RUN_TESTS = false` → Manual mode
- ✅ `TEST_FULL_SOLVER = false` → LU factorization only
- ✅ `TEST_FULL_SOLVER = true` → Full system solving

**Just change the flags and run!** 🚀

```bash
julia midterm_p03.jl
```

---

📖 **For more details, see `USAGE.md`**  
📋 **For file structure info, see `README.md`**
