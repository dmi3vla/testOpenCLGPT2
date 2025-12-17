// OpenCL Kernels for Matrix Multiplication Test
// Target: AMD GLADIUS (gfx701) via Mesa/Clover/LLVM
// Simplified for Clover compatibility

// FMA Stress Test Kernel - для разогрева GPU
// Clover не всегда поддерживает fma(), используем обычные операции
__kernel void fma_stress(__global float *a, 
                         __global float *b, 
                         __global float *c,
                         const unsigned int n,
                         const unsigned int iterations) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    
    float val_a = a[gid];
    float val_b = b[gid];
    float val_c = c[gid];
    
    // Выполняем iterations операций (замена FMA на обычные операции)
    for (unsigned int i = 0; i < iterations; i++) {
        val_c = val_a * val_b + val_c;
        val_c = val_a * val_b + val_c;
        val_c = val_a * val_b + val_c;
        val_c = val_a * val_b + val_c;
    }
    
    c[gid] = val_c;
}

// Naive Matrix Multiplication Kernel
// C = A * B, где все матрицы size x size
__kernel void matrix_multiply(__global const float *A,
                              __global const float *B, 
                              __global float *C,
                              const unsigned int size) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row >= size || col >= size) return;
    
    float sum = 0.0f;
    for (unsigned int k = 0; k < size; k++) {
        sum += A[row * size + k] * B[k * size + col];
    }
    
    C[row * size + col] = sum;
}

__kernel void matmul_a_bt(__global const float *A,
                          __global const float *B_T,
                          __global float *C,
                          const unsigned int M,
                          const unsigned int N,
                          const unsigned int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if ((unsigned int)row >= M || (unsigned int)col >= N) return;

    float sum = 0.0f;
    const unsigned int a_off = ((unsigned int)row) * K;
    const unsigned int b_off = ((unsigned int)col) * K;
    for (unsigned int k = 0; k < K; k++) {
        sum += A[a_off + k] * B_T[b_off + k];
    }
    C[((unsigned int)row) * N + (unsigned int)col] = sum;
}
