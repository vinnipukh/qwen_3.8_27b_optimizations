// bench_bp.cpp — Pinned host<->device transfer bandwidth over PCIe (B_P probe)
//
// FreeToken paper §3.2: B_P = measured host-to-device expert-transfer
// bandwidth (what a cache fill costs). On this Windows box we measure through
// OpenCL (Adrenalin ICD), which is the same WDDM/DMA path a HIP engine would
// use for clEnqueueWriteBuffer-style fills on Linux ROCm.
//
// Methodology mirrors clpeak: large non-blocking writes/reads + clFinish,
// multiple reps, median reported. Sizes swept so you can see where the link
// saturates and driver overhead stops mattering.
//
// Build: zig c++ -O3 -o ../bin/bench_bp.exe bench_bp.cpp
// Run:   ./bin/bench_bp.exe [sizes_mb_csv] [reps]
//        e.g. ./bin/bench_bp.exe 16,64,256 10

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

// ---- minimal OpenCL dynamic bindings (x64: default calling conv is fine) ----
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint  cl_bool;
typedef void*    cl_platform_id;
typedef void*    cl_device_id;
typedef void*    cl_context;
typedef void*    cl_command_queue;
typedef void*    cl_mem;
typedef void*    cl_event;
typedef cl_uint  cl_device_type;

#define CL_SUCCESS                    0
#define CL_DEVICE_TYPE_GPU            (1 << 2)
#define CL_DEVICE_NAME                0x102B
#define CL_DEVICE_VERSION             0x1003
#define CL_DEVICE_GLOBAL_MEM_SIZE     0x101F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE  0x1010
#define CL_TRUE                       1
#define CL_FALSE                      0
#define CL_MEM_READ_WRITE             (1 << 2)
#define CL_MEM_ALLOC_HOST_PTR         (1 << 6)

struct ClApi {
    HMODULE dll = nullptr;
    cl_int  (*GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
    cl_int  (*GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*) = nullptr;
    cl_int  (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*) = nullptr;
    cl_context (*CreateContext)(const intptr_t*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*) = nullptr;
    cl_command_queue (*CreateQueueWithProps)(cl_context, cl_device_id, const cl_ulong*, cl_int*) = nullptr;
    cl_command_queue (*CreateQueueLegacy)(cl_context, cl_device_id, cl_ulong, cl_int*) = nullptr;
    cl_mem   (*CreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*) = nullptr;
    cl_int   (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int   (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int   (*Finish)(cl_command_queue) = nullptr;
    cl_int   (*ReleaseMemObject)(cl_mem) = nullptr;
    cl_int   (*ReleaseCommandQueue)(cl_command_queue) = nullptr;
    cl_int   (*ReleaseContext)(cl_context) = nullptr;

    bool load() {
        dll = LoadLibraryA("OpenCL.dll");
        if (!dll) { fprintf(stderr, "OpenCL.dll not found\n"); return false; }
        auto sym = [&](const char* n) { return (void*)GetProcAddress(dll, n); };
        GetPlatformIDs      = decltype(GetPlatformIDs)((void*)sym("clGetPlatformIDs"));
        GetDeviceIDs        = decltype(GetDeviceIDs)((void*)sym("clGetDeviceIDs"));
        GetDeviceInfo       = decltype(GetDeviceInfo)((void*)sym("clGetDeviceInfo"));
        CreateContext       = decltype(CreateContext)((void*)sym("clCreateContext"));
        CreateQueueWithProps= decltype(CreateQueueWithProps)((void*)sym("clCreateCommandQueueWithProperties"));
        CreateQueueLegacy   = decltype(CreateQueueLegacy)((void*)sym("clCreateCommandQueue"));
        CreateBuffer        = decltype(CreateBuffer)((void*)sym("clCreateBuffer"));
        EnqueueWriteBuffer  = decltype(EnqueueWriteBuffer)((void*)sym("clEnqueueWriteBuffer"));
        EnqueueReadBuffer   = decltype(EnqueueReadBuffer)((void*)sym("clEnqueueReadBuffer"));
        Finish              = decltype(Finish)((void*)sym("clFinish"));
        ReleaseMemObject    = decltype(ReleaseMemObject)((void*)sym("clReleaseMemObject"));
        ReleaseCommandQueue = decltype(ReleaseCommandQueue)((void*)sym("clReleaseCommandQueue"));
        ReleaseContext      = decltype(ReleaseContext)((void*)sym("clReleaseContext"));
        return GetPlatformIDs && GetDeviceIDs && GetDeviceInfo && CreateContext &&
               CreateBuffer && EnqueueWriteBuffer && EnqueueReadBuffer && Finish;
    }
};

static void print_dev_info(ClApi& cl, cl_device_id dev, const char* label, cl_uint param) {
    char buf[512] = {0};
    if (cl.GetDeviceInfo(dev, param, sizeof(buf) - 1, buf, nullptr) == CL_SUCCESS)
        printf("%s: %s\n", label, buf);
}

struct Stats { double min, med, max; };
static Stats summarize(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return { v.front(), v[v.size() / 2], v.back() };
}

int main(int argc, char** argv) {
    std::string sizes_csv = (argc > 1) ? argv[1] : "16,64,256";
    int reps = (argc > 2) ? atoi(argv[2]) : 10;

    ClApi cl;
    if (!cl.load()) return 1;

    cl_platform_id plat = nullptr;
    cl_uint nplat = 0;
    if (cl.GetPlatformIDs(1, &plat, &nplat) != CL_SUCCESS || nplat == 0) {
        fprintf(stderr, "no OpenCL platform\n"); return 1;
    }
    cl_device_id dev = nullptr;
    cl_uint ndev = 0;
    if (cl.GetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &ndev) != CL_SUCCESS || ndev == 0) {
        fprintf(stderr, "no OpenCL GPU device\n"); return 1;
    }

    printf("== bench_bp: PCIe transfer bandwidth (B_P probe) ==\n");
    print_dev_info(cl, dev, "device",       CL_DEVICE_NAME);
    print_dev_info(cl, dev, "driver spec",  CL_DEVICE_VERSION);
    cl_ulong memsz = 0, maxalloc = 0;
    cl.GetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memsz), &memsz, nullptr);
    cl.GetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxalloc), &maxalloc, nullptr);
    printf("VRAM: %.1f GiB | max single alloc: %.0f MiB\n",
           memsz / 1073741824.0, maxalloc / 1048576.0);

    cl_int err = 0;
    cl_context ctx = cl.CreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "CreateContext err %d\n", err); return 1; }

    cl_command_queue q = nullptr;
    if (cl.CreateQueueWithProps) {
        cl_ulong props[] = { 0 };   // in-order, no profiling
        q = cl.CreateQueueWithProps(ctx, dev, props, &err);
    }
    if (!q && cl.CreateQueueLegacy) q = cl.CreateQueueLegacy(ctx, dev, 0, &err);
    if (!q) { fprintf(stderr, "queue creation failed (%d)\n", err); return 1; }

    // parse sizes
    std::vector<size_t> sizes_mb;
    {
        std::string s = sizes_csv; size_t pos = 0;
        while (pos <= s.size()) {
            size_t comma = s.find(',', pos);
            std::string tok = s.substr(pos, comma == std::string::npos ?
                                       std::string::npos : comma - pos);
            if (!tok.empty()) sizes_mb.push_back((size_t)atoi(tok.c_str()));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }

    LARGE_INTEGER freq; QueryPerformanceFrequency(&freq);

    printf("\n%-4s %-8s %-10s %-10s %-10s\n", "dir", "MiB", "min_GB/s", "med_GB/s", "max_GB/s");
    for (size_t mb : sizes_mb) {
        size_t bytes = mb * 1024ull * 1024ull;
        if (bytes > maxalloc) { printf("skip %zu MiB > max alloc\n", mb); continue; }

        // page-locked-ish host buffer (VirtualAlloc large pages not required;
        // the driver pins pages for the DMA window during each enqueue)
        void* host = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!host) { fprintf(stderr, "host alloc failed\n"); return 1; }
        memset(host, 0x5A, bytes);

        cl_mem dbuf = cl.CreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "CreateBuffer err %d\n", err); return 1; }

        struct Dir { const char* name; bool write; };
        Dir dirs[2] = { {"h2d", true}, {"d2h", false} };

        for (auto& d : dirs) {
            // warmup x2
            for (int w = 0; w < 2; ++w) {
                if (d.write) cl.EnqueueWriteBuffer(q, dbuf, CL_FALSE, 0, bytes, host, 0, nullptr, nullptr);
                else         cl.EnqueueReadBuffer (q, dbuf, CL_FALSE, 0, bytes, host, 0, nullptr, nullptr);
                cl.Finish(q);
            }
            std::vector<double> gbps;
            for (int r = 0; r < reps; ++r) {
                LARGE_INTEGER t0, t1;
                QueryPerformanceCounter(&t0);
                if (d.write) cl.EnqueueWriteBuffer(q, dbuf, CL_FALSE, 0, bytes, host, 0, nullptr, nullptr);
                else         cl.EnqueueReadBuffer (q, dbuf, CL_FALSE, 0, bytes, host, 0, nullptr, nullptr);
                cl.Finish(q);
                QueryPerformanceCounter(&t1);
                double sec = (t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
                gbps.push_back((double)bytes / sec / 1e9);
            }
            Stats st = summarize(gbps);
            printf("%-4s %-8zu %-10.1f %-10.1f %-10.1f\n", d.name, mb, st.min, st.med, st.max);
            printf("RESULT dir=%s mb=%zu med_gbps=%.1f\n", d.name, mb, st.med);
        }
        cl.ReleaseMemObject(dbuf);
        VirtualFree(host, 0, MEM_RELEASE);
    }
    cl.ReleaseCommandQueue(q);
    cl.ReleaseContext(ctx);
    printf("\nNote: WDDM adds per-enqueue overhead; large transfers approximate the\n");
    printf("steady-state DMA rate. Native Linux/ROCm typically reports slightly higher.\n");
    return 0;
}
