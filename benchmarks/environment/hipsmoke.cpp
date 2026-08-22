// hipsmoke: minimal device-execution proof for gfx1100 under ROCDXG
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void bump(int* p) { *p += 1; }

int main() {
    int h = -1;
    int* d = nullptr;
    if (hipMalloc(&d, 4) != hipSuccess) { printf("HIPMALLOC-FAIL\n"); return 1; }
    hipMemset(d, 0, 4);
    bump<<<1, 1>>>(d);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("SYNC-FAIL:%d\n", (int)err); return 3; }
    hipMemcpy(&h, d, 4, hipMemcpyDeviceToHost);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("RESULT=%d ARCH=%s NAME=%s\n", h, prop.gcnArchName, prop.name);
    hipFree(d);
    return (h == 1) ? 0 : 2;
}
