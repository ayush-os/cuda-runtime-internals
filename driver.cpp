#include <cuda.h>
#include <iostream>
#include <vector>

// Helper for error checking (The Driver API returns CUresult)
#define CHECK_DRV(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(CUresult code, const char *file, int line) {
   if (code != CUDA_SUCCESS) {
      const char* errStr;
      cuGetErrorString(code, &errStr);
      fprintf(stderr,"Driver Error: %s %s %d\n", errStr, file, line);
      exit(code);
   }
}

int main() {
    // 1. INITIALIZE DRIVER
    cuInit(0);

    // 2. GET DEVICE HANDLE
    CUdevice device;
    cuDeviceGet(&device, 0);

    // 3. CONTEXT MANAGEMENT
    CUcontext ctxA, ctxB;
    cuCtxCreate(&ctxA, 0, device);
    cuCtxCreate(&ctxB, 0, device);

    // 4. LOAD MODULE & FUNCTION
    CUmodule hModule;
    CUfunction hKernel;
    CUfunction vecAdd;
    cuModuleLoad(&hModule, "vector_add.ptx");
    cuModuleGetFunction(&vecAdd, hModule, "vector_add");

    int N = 1024;
    size_t size = N * sizeof(float);
    CUdeviceptr d_out, d_a, d_b;
    cuMemAlloc(&d_a, size);
    cuMemAlloc(&d_b, size);
    cuMemAlloc(&d_out, size);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

   for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    cuMemcpyHtoD(d_a, h_A, size);
    cuMemcpyHtoD(d_b, h_B, size);

    // 6. KERNEL LAUNCH
    // Driver API launches require an array of pointers to the arguments
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &d_out, &d_a, &d_b, &N };

    // TODO: Launch the kernel
    // Hint: cuLaunchKernel(hKernel, gridDimX, 1, 1, blockDimX, 1, 1, 0, 0, args, 0);
    cuLaunchKernel(hKernel,
                   blocksPerGrid, 1, 1, threadsPerBlock, 1, 1,
                   0, 0, args, 0);

   

    // 7. CLEANUP
    // TODO: Destroy contexts, unload module.
    
    return 0;
}