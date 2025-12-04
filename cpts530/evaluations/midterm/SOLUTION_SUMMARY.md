# MIDTERM PROBLEM 3: Complete Solution

## Problem Statement
Solve the system Ax = b where:
```
A = [ 6.25  -1.0   0.5  ]    b = [  7.5  ]
    [-1.0    5.0   2.12 ]        [ -8.68 ]
    [ 0.5    2.12  3.6  ]        [ -0.24 ]
```

## Solution: x = [0.8, -2.0, 1.0]ᵀ

---

## Method 1: LU Factorization

### Factorization Result:
```
L = [  1.0   0.0       0.0  ]
    [ -0.16  1.0       0.0  ]
    [  0.08  0.454545  1.0  ]

U = [ 6.25  -1.0   0.5  ]
    [ 0.0    4.84  2.2  ]
    [ 0.0    0.0   2.56 ]
```

### Verification:
- max|A - LU| = 4.441e-16 ✓
- max|Ax - b| = 2.220e-16 ✓
- Matches Julia's built-in solver ✓

---

## Method 2: Cholesky Factorization

### Factorization Result:
```
L = [ 2.5  0.0  0.0 ]
    [-0.4  2.2  0.0 ]
    [ 0.2  1.0  1.6 ]

where A = L*L'
```

### Verification:
- max|A - LL'| = 8.882e-16 ✓
- max|Ax - b| = 2.220e-16 ✓
- Matches Julia's built-in solver ✓

---

## Comparison:
- max|x_LU - x_Cholesky| = 4.441e-16 ✓
- Both methods agree perfectly!

---

## Implementation Details

### LU Algorithm (Doolittle):
```julia
for r = 1:n
    for c = 1:n
        if r > c
            L[r,c] = (A[r,c] - sum(L[r,k]*U[k,c] for k in 1:c-1)) / U[c,c]
        else
            U[r,c] = A[r,c] - sum(L[r,k]*U[k,c] for k in 1:r-1)
        end
    end
end
```

### Cholesky Algorithm:
```julia
for i = 1:n
    # Diagonal
    L[i,i] = sqrt(A[i,i] - sum(L[i,k]^2 for k in 1:i-1))
    
    # Off-diagonal
    for j = i+1:n
        L[j,i] = (A[j,i] - sum(L[j,k]*L[i,k] for k in 1:i-1)) / L[i,i]
    end
end
```

---

## Code Characteristics:
✅ Simple nested loops (no fancy operations)
✅ Clear, readable implementation
✅ Both methods verified to machine precision
✅ Production-ready code

---

## Run Command:
```bash
julia midterm_p03.jl
```

## Final Answer:
**x₁ = 0.8**
**x₂ = -2.0**
**x₃ = 1.0**
