# NumPy Revision Repository

A comprehensive NumPy learning resource covering fundamentals to advanced concepts.

## Repository Structure

### Module 01: NumPy Basics
| Notebook | Topics |
|----------|--------|
| [01_introduction_to_numpy](01_numpy_basics/01_introduction_to_numpy.ipynb) | What is NumPy, comparison with lists, installation |
| [02_array_creation](01_numpy_basics/02_array_creation.ipynb) | zeros, ones, arange, linspace, random |
| [03_array_attributes_and_dtypes](01_numpy_basics/03_array_attributes_and_dtypes.ipynb) | shape, dtype, ndim, itemsize |
| [04_indexing_and_slicing](01_numpy_basics/04_indexing_and_slicing.ipynb) | Basic indexing, slicing, views |

### Module 02: Array Manipulation
| Notebook | Topics |
|----------|--------|
| [01_reshape_and_resize](02_array_manipulation/01_reshape_and_resize.ipynb) | reshape, flatten, ravel, resize |
| [02_concatenate_and_split](02_array_manipulation/02_concatenate_and_split.ipynb) | concatenate, split, insert, delete |
| [03_stacking_and_tiling](02_array_manipulation/03_stacking_and_tiling.ipynb) | vstack, hstack, tile, repeat |
| [04_transposing_and_swapping](02_array_manipulation/04_transposing_and_swapping.ipynb) | transpose, swapaxes, flip, roll |

### Module 03: Mathematical Operations
| Notebook | Topics |
|----------|--------|
| [01_arithmetic_operations](03_mathematical_operations/01_arithmetic_operations.ipynb) | Element-wise ops, ufuncs, aggregations |
| [02_statistical_operations](03_mathematical_operations/02_statistical_operations.ipynb) | mean, std, variance, percentiles |
| [03_linear_algebra](03_mathematical_operations/03_linear_algebra.ipynb) | Matrix ops, dot, solve, eigenvalues |
| [04_trigonometric_and_exponential](03_mathematical_operations/04_trigonometric_and_exponential.ipynb) | sin, cos, exp, log, special functions |

### Module 04: Broadcasting and Vectorization
| Notebook | Topics |
|----------|--------|
| [01_broadcasting_fundamentals](04_broadcasting_and_vectorization/01_broadcasting_fundamentals.ipynb) | Broadcasting rules and patterns |
| [02_vectorization_techniques](04_broadcasting_and_vectorization/02_vectorization_techniques.ipynb) | Replacing loops, np.vectorize |
| [03_avoiding_loops](04_broadcasting_and_vectorization/03_avoiding_loops.ipynb) | Advanced patterns, einsum |

### Module 05: Advanced Indexing
| Notebook | Topics |
|----------|--------|
| [01_fancy_indexing](05_advanced_indexing/01_fancy_indexing.ipynb) | Integer array indexing, np.ix_ |
| [02_boolean_indexing](05_advanced_indexing/02_boolean_indexing.ipynb) | Boolean masks, np.where |
| [03_advanced_slicing_techniques](05_advanced_indexing/03_advanced_slicing_techniques.ipynb) | Views, strides, structured arrays |

### Module 06: File I/O
| Notebook | Topics |
|----------|--------|
| [01_saving_and_loading_arrays](06_file_io/01_saving_and_loading_arrays.ipynb) | save, load, savez, memmap |
| [02_working_with_text_files](06_file_io/02_working_with_text_files.ipynb) | loadtxt, genfromtxt, savetxt |
| [03_binary_and_compressed_files](06_file_io/03_binary_and_compressed_files.ipynb) | Binary formats, HDF5, MATLAB |

### Module 07: Performance Optimization
| Notebook | Topics |
|----------|--------|
| [01_memory_layout_and_views](07_performance_optimization/01_memory_layout_and_views.ipynb) | C/F order, strides, contiguity |
| [02_vectorization_best_practices](07_performance_optimization/02_vectorization_best_practices.ipynb) | Optimization patterns and pitfalls |
| [03_profiling_and_benchmarking](07_performance_optimization/03_profiling_and_benchmarking.ipynb) | timeit, cProfile, memory profiling |

### Module 08: Practice Problems
| Notebook | Topics |
|----------|--------|
| [01_beginner_exercises](08_practice_problems/01_beginner_exercises.ipynb) | 10 fundamental exercises |
| [02_intermediate_exercises](08_practice_problems/02_intermediate_exercises.ipynb) | 10 intermediate challenges |
| [03_advanced_exercises](08_practice_problems/03_advanced_exercises.ipynb) | 10 advanced problems (KNN, PCA, FFT) |
| [04_interview_questions](08_practice_problems/04_interview_questions.ipynb) | 20 common interview questions |

## Quick Start

```python
import numpy as np

# Create array
arr = np.array([1, 2, 3, 4, 5])

# Basic operations
print(arr.mean())      # 3.0
print(arr.sum())       # 15
print(arr * 2)         # [2, 4, 6, 8, 10]
```

## Learning Path

1. **Beginners**: Start with Module 01-02
2. **Intermediate**: Progress to Module 03-05
3. **Advanced**: Complete Module 06-07
4. **Interview Prep**: Practice with Module 08

## Key Concepts Reference

| Concept | Description |
|---------|-------------|
| **Broadcasting** | Automatic expansion for compatible shapes |
| **Vectorization** | Replace loops with array operations |
| **Views** | Memory reference without copy |
| **Fancy Indexing** | Integer array indexing (creates copy) |
| **Strides** | Bytes to jump per dimension |

## Requirements

- Python 3.7+
- NumPy 1.20+

```bash
pip install numpy
```

## Total Content

- **8 Modules**
- **28 Notebooks**
- **100+ Code Examples**
- **50+ Practice Problems**
