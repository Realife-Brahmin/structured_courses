# MIDTERM PROBLEM 3 - LU Factorization Results

## Problem Statement

Given the system Ax = b where:

$$\mathbf{A} = \begin{bmatrix} 6.25 & -1 & 0.5 \\ -1 & 5 & 2.12 \\ 0.5 & 2.12 & 3.6 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 7.5 \\ -8.68 \\ -0.24 \end{bmatrix}$$

---

## LU Factorization Results

### Matrix L (Lower Triangular with 1's on diagonal):
```
L = [  1.0        0.0         0.0      ]
    [ -0.16       1.0         0.0      ]
    [  0.08       0.454545    1.0      ]
```

More precisely:
- L[1,1] = 1.0
- L[2,1] = -0.16 = -1/6.25 = -4/25
- L[3,1] = 0.08 = 0.5/6.25 = 2/25
- L[2,2] = 1.0
- L[3,2] = 0.454545... = 5/11
- L[3,3] = 1.0

### Matrix U (Upper Triangular):
```
U = [ 6.25      -1.0        0.5       ]
    [ 0.0        4.84       2.2       ]
    [ 0.0        0.0        2.56      ]
```

More precisely:
- U[1,1] = 6.25 = 25/4
- U[1,2] = -1.0
- U[1,3] = 0.5 = 1/2
- U[2,2] = 4.84 = 121/25
- U[2,3] = 2.2 = 11/5
- U[3,3] = 2.56 = 64/25

---

## Verification

**A = L × U?**

```julia
L * U = [  6.25  -1.0   0.5  ]
        [ -1.0    5.0   2.12 ]
        [  0.5    2.12  3.6  ]
```

**Error: A - L×U:**
```
Error = [ 0.0  0.0  0.0            ]
        [ 0.0  0.0  0.0            ]
        [ 0.0  0.0  4.44089e-16    ]  ← Machine precision error only!
```

**Max absolute error: 4.441e-16** ✅

✅ **LU factorization is CORRECT!** (error is at machine precision level)

---

## Next Steps

1. **Solve Ly = b** using forward substitution:
   ```
   y₁ = 7.5 / 1.0 = 7.5
   y₂ = (-8.68 - (-0.16)(7.5)) / 1.0 = -7.48
   y₃ = (-0.24 - (0.08)(7.5) - (0.454545)(-7.48)) / 1.0 = 2.56
   ```

2. **Solve Ux = y** using backward substitution:
   ```
   x₃ = 2.56 / 2.56 = 1.0
   x₂ = ((-7.48) - (2.2)(1.0)) / 4.84 = -2.0
   x₁ = (7.5 - (-1.0)(-2.0) - (0.5)(1.0)) / 6.25 = 0.8
   ```

3. **Expected solution:**
   ```
   x = [0.8, -2.0, 1.0]ᵀ
   ```

---

## Implementation Status

- ✅ `get_LU_factors(A)` - IMPLEMENTED and verified
- ⏳ `solve_Ly_is_equal_to_b(L, b)` - TODO
- ⏳ `solve_Ux_is_equal_to_y(U, y)` - TODO
- ⏳ Full system solving test - Pending above functions

---

## How to Run

```bash
cd cpts530/midterm
julia midterm_p03.jl
```

The actual midterm matrix is now in **TEST 2**.

To test full system solving after implementing forward/backward substitution:
```julia
const TEST_FULL_SOLVER = true  # Change this in midterm_p03.jl
```

Then run `test_problem_1()` which now uses the actual midterm problem!
