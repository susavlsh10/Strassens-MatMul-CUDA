# Strassens-MatMul-CUDA

**Title: Parallelizing Strassen's Matrix-Multiplication Algorithm with CUDA**

**Project Overview:**
This project aims to implement and parallelize Strassen's Matrix-Multiplication Algorithm using CUDA, a high-performance parallel computing platform and application programming interface (API). The algorithm efficiently computes the product matrix C from two matrices A and B, each of size n x n. The standard algorithm for matrix multiplication has a time complexity of O(n^3), while Strassen's algorithm reduces it to approximately O(n^2.8) operations.

Standard algorithm

```python
for i in range(1, n + 1):
    for j in range(1, n + 1):
        for k in range(1, n + 1):
            𝐶𝑖𝑗 += 𝐴𝑖𝑘 * 𝐵𝑘𝑗
```

**Strassen's Algorithm:**
Strassen's algorithm divides matrices A, B, and C into equal-sized blocks, leading to a recursive approach for matrix multiplication. The product matrix C is computed using seven matrix multiplications and eighteen matrix additions, significantly reducing the number of multiplicative operations compared to the standard algorithm.

![Strassen's Algorithm](https://github.com/susavlsh10/Strassens-MatMul-CUDA/blob/main/algorithm.png)




**Project Implementation:**
This project will implement Strassen's Matrix-Multiplication Algorithm using CUDA, leveraging the parallel processing power of GPUs. CUDA will be used to optimize the algorithm's performance by parallelizing the matrix multiplication and addition operations. The project will also consider terminating the recursion at a specific level (k') to handle smaller matrix sizes using the standard matrix multiplication approach.

The primary goals of this project are to:
- Implement Strassen's algorithm in CUDA for parallel matrix multiplication.
- Optimize the algorithm for large matrix sizes.
- Explore and measure the performance improvements achieved through parallelization.
- Provide a clear and well-documented codebase for others to understand and use.

This project aims to harness the power of GPUs to enhance the efficiency of matrix multiplication and may serve as a valuable tool for various scientific and computational applications.
