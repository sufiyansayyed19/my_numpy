# NumPy Cheat Sheet

## Array Creation

```python
np.array([1, 2, 3])              # From list
np.zeros((3, 4))                 # All zeros
np.ones((3, 4))                  # All ones
np.empty((3, 4))                 # Uninitialized
np.full((3, 4), 7)               # Fill with value
np.eye(3)                        # Identity matrix
np.arange(0, 10, 2)              # Range with step
np.linspace(0, 1, 5)             # N evenly spaced
np.random.rand(3, 4)             # Uniform [0, 1)
np.random.randn(3, 4)            # Normal (0, 1)
np.random.randint(0, 10, (3,4))  # Random integers
```

## Array Attributes

```python
arr.shape      # Dimensions tuple
arr.ndim       # Number of dimensions
arr.size       # Total elements
arr.dtype      # Data type
arr.itemsize   # Bytes per element
arr.nbytes     # Total bytes
arr.T          # Transpose
```

## Reshaping

```python
arr.reshape(3, 4)        # New shape (view)
arr.flatten()            # 1D copy
arr.ravel()              # 1D view
arr.squeeze()            # Remove size-1 dims
arr[:, np.newaxis]       # Add dimension
np.expand_dims(arr, 0)   # Add axis
```

## Indexing & Slicing

```python
arr[0]                   # First element
arr[-1]                  # Last element
arr[1:4]                 # Slice [1,4)
arr[::2]                 # Every 2nd
arr[::-1]                # Reverse
arr[1, 2]                # 2D index
arr[:, 0]                # First column
arr[arr > 5]             # Boolean mask
arr[[0, 2, 4]]           # Fancy index
```

## Stacking & Splitting

```python
np.concatenate([a, b])        # Join arrays
np.vstack([a, b])             # Stack vertically
np.hstack([a, b])             # Stack horizontally
np.stack([a, b], axis=0)      # New axis
np.split(arr, 3)              # Split into 3
np.array_split(arr, 3)        # Unequal splits OK
```

## Math Operations

```python
# Element-wise
a + b, a - b, a * b, a / b
a ** 2                   # Power
np.sqrt(arr)             # Square root
np.exp(arr)              # Exponential
np.log(arr)              # Natural log
np.sin(arr)              # Trig functions

# Aggregations
arr.sum()                # Total
arr.mean()               # Average
arr.std()                # Std deviation
arr.min(), arr.max()     # Extremes
arr.argmin(), arr.argmax()  # Index of extremes
arr.sum(axis=0)          # Sum along axis
np.cumsum(arr)           # Cumulative sum
```

## Linear Algebra

```python
np.dot(a, b)             # Dot product
a @ b                    # Matrix multiply
np.linalg.inv(a)         # Inverse
np.linalg.det(a)         # Determinant
np.linalg.eig(a)         # Eigenvalues
np.linalg.svd(a)         # SVD
np.linalg.solve(A, b)    # Solve Ax = b
np.linalg.norm(a)        # Norm
```

## Statistics

```python
np.mean(arr)             # Mean
np.median(arr)           # Median
np.std(arr)              # Standard deviation
np.var(arr)              # Variance
np.percentile(arr, 75)   # Percentile
np.corrcoef(a, b)        # Correlation
np.histogram(arr, bins=10)  # Histogram
```

## Comparison & Logic

```python
arr > 5                  # Boolean array
np.where(arr > 5)        # Indices where True
np.where(arr > 5, a, b)  # Conditional select
np.any(arr > 5)          # Any True?
np.all(arr > 5)          # All True?
np.isin(arr, [1,2,3])    # Check membership
np.unique(arr)           # Unique values
```

## Sorting

```python
np.sort(arr)             # Sorted copy
arr.sort()               # In-place sort
np.argsort(arr)          # Sort indices
np.lexsort((b, a))       # Multi-key sort
np.partition(arr, 3)     # Partial sort
```

## Set Operations

```python
np.unique(arr)           # Unique elements
np.intersect1d(a, b)     # Common elements
np.union1d(a, b)         # All unique
np.setdiff1d(a, b)       # In a, not in b
np.in1d(a, b)            # a elements in b?
```

## Broadcasting Rules

```
Shape A    Shape B    Result
(3, 4)  +  (4,)    =  (3, 4)
(3, 1)  +  (1, 4)  =  (3, 4)
(3, 4)  +  (1,)    =  (3, 4)
(5,3,1) +  (3, 4)  =  (5, 3, 4)
```

## File I/O

```python
# Binary
np.save('file.npy', arr)
np.load('file.npy')
np.savez('file.npz', a=arr1, b=arr2)
np.savez_compressed('file.npz', arr)

# Text
np.savetxt('file.csv', arr, delimiter=',')
np.loadtxt('file.csv', delimiter=',')
np.genfromtxt('file.csv', delimiter=',')  # Handles missing
```

## Random Numbers

```python
np.random.seed(42)            # Set seed
np.random.rand(3, 4)          # Uniform [0, 1)
np.random.randn(3, 4)         # Normal (0, 1)
np.random.randint(0, 10, 5)   # Random ints
np.random.choice(arr, 3)      # Random sample
np.random.shuffle(arr)        # Shuffle in-place
np.random.permutation(arr)    # Shuffled copy
```

## Views vs Copies

```python
# VIEWS (share memory)
arr[1:5]          # Slicing
arr.reshape(2,3)  # Reshape
arr.T             # Transpose
arr.ravel()       # Flatten (if contiguous)

# COPIES (independent)
arr[[1,3,5]]      # Fancy indexing
arr[arr > 5]      # Boolean indexing
arr.flatten()     # Always copy
arr.copy()        # Explicit copy
```

## Performance Tips

```python
# Use vectorized ops
arr * 2                   # Good
[x*2 for x in arr]        # Bad

# Preallocate
result = np.empty(1000)   # Good
result = []               # Bad (append loop)

# In-place operations
arr *= 2                  # No allocation
np.add(a, b, out=result)  # Reuse array

# Check contiguity
np.ascontiguousarray(arr)
```

## Common Patterns

```python
# Normalize
(arr - arr.mean()) / arr.std()

# One-hot encoding
np.eye(n_classes)[labels]

# Moving average
np.convolve(arr, np.ones(k)/k, 'valid')

# Pairwise distances
np.sqrt(((a[:,None,:] - b[None,:,:])**2).sum(2))

# Softmax
exp_x = np.exp(x - x.max())
exp_x / exp_x.sum()
```

## Quick Reference

| Operation | Function |
|-----------|----------|
| Create zeros | `np.zeros(shape)` |
| Create range | `np.arange(start, stop, step)` |
| Reshape | `arr.reshape(shape)` |
| Transpose | `arr.T` or `np.transpose(arr)` |
| Sum | `arr.sum(axis=None)` |
| Mean | `arr.mean(axis=None)` |
| Max/Min | `arr.max()`, `arr.min()` |
| Sort | `np.sort(arr)` |
| Unique | `np.unique(arr)` |
| Where | `np.where(cond, x, y)` |
| Dot | `np.dot(a, b)` or `a @ b` |
| Stack | `np.vstack`, `np.hstack` |
| Save | `np.save(file, arr)` |
| Load | `np.load(file)` |
