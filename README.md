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

Consider the following partitioning of the matrices into equal-sized blocks:

𝐴 = [𝐴11 𝐴12]
     [𝐴21 𝐴22]

𝐵 = [𝐵11 𝐵12]
     [𝐵21 𝐵22]

𝐶 = [𝐶11 𝐶12]
     [𝐶21 𝐶22]   (1)

This partitioning allows for a straightforward approach to compute C:

𝐶11 = 𝐴11𝐵11 + 𝐴12𝐵21
𝐶12 = 𝐴11𝐵12 + 𝐴12𝐵22
𝐶21 = 𝐴21𝐵11 + 𝐴22𝐵21
𝐶22 = 𝐴21𝐵12 + 𝐴22𝐵22   (2)

This approach requires 8 multiplications and 4 additions among matrices of size n/2 x n/2. Strassen’s algorithm computes the product as:

𝐶11 = 𝑀1 + 𝑀4 - 𝑀5 + 𝑀7
𝐶12 = 𝑀3 + 𝑀5
𝐶21 = 𝑀2 + 𝑀4
𝐶22 = 𝑀1 - 𝑀2 + 𝑀3 + 𝑀6  (3)

Where:

𝑀1 = (𝐴11 + 𝐴22)(𝐵11 + 𝐵22)
𝑀2 = (𝐴21 + 𝐴22)𝐵11
𝑀3 = 𝐴11(𝐵12 - 𝐵22)
𝑀4 = 𝐴22(𝐵21 - 𝐵11)
𝑀5 = (𝐴11 + 𝐴12)𝐵22
𝑀6 = (𝐴21 - 𝐴11)(𝐵11 + 𝐵12)
𝑀7 = (𝐴12 - 𝐴22)(𝐵21 + 𝐵22)  (4)




**Project Implementation:**
This project will implement Strassen's Matrix-Multiplication Algorithm using CUDA, leveraging the parallel processing power of GPUs. CUDA will be used to optimize the algorithm's performance by parallelizing the matrix multiplication and addition operations. The project will also consider terminating the recursion at a specific level (k') to handle smaller matrix sizes using the standard matrix multiplication approach.

The primary goals of this project are to:
- Implement Strassen's algorithm in CUDA for parallel matrix multiplication.
- Optimize the algorithm for large matrix sizes.
- Explore and measure the performance improvements achieved through parallelization.
- Provide a clear and well-documented codebase for others to understand and use.

This project aims to harness the power of GPUs to enhance the efficiency of matrix multiplication and may serve as a valuable tool for various scientific and computational applications.
