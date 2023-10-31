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


## Compilation and Execution

To compile the code, follow these steps:

```bash
make

./MatMul.exe <k> <k'>

```
The size of the matrix (n x n) is determined by n = 2<sup>k</sup>, and the size of the terminal matrix (s x s) is determined by s = n /2<sup>k'</sup>

![Speedup vs k](https://github.com/susavlsh10/Strassens-MatMul-CUDA/blob/main/speedupvsk.png)

The figure above illustrates the speedup achieved by the CUDA-based Strassen's algorithm in comparison to the CPU-based algorithm for different values of k with k'= 1. It is evident that the speedup remains below 1 for matrix sizes less than 32x32, indicating that the CPU-based algorithm outperforms the GPU-based algorithm in such scenarios. This is primarily because the overhead cost of data transfer becomes significant compared to the actual processing time for smaller matrices. In contrast, the CPU-based algorithm does not require data transfer since all data is already in the CPU memory. However, as the matrix size surpasses this threshold, the difference between the CPU and GPU implementations becomes noticeable for parallel matrix multiplication. The speedup increases as larger matrices can leverage the parallel hardware capabilities of the GPU more effectively. For instance, the speedup started at 4.928336x for k=6 and increased all the way to 5364.372x for k=12.
