// bench_bh.cpp — Host-memory streaming bandwidth probe (FreeToken paper's B_H)
//
// Paper §3.2: B_H = measured effective bandwidth of the CPU-side MoE expert
// kernel, i.e. how fast the CPU can stream expert weights out of DRAM while
// doing light math (dequant + multiply-accumulate). Memory-bound at batch 1.
//
// We simulate that with an AVX2 FMA streaming pass over a buffer much larger
// than LLC (default 2 GiB >> 32 MB L3 on a Ryzen 7 5700X), multi-threaded.
// Two modes:
//   read : pure streaming read (bandwidth ceiling)
//   fma  : read + FMAs (MoE-expert-like work per byte)
//
// Build: zig c++ -O3 -mavx2 -mfma -o ../bin/bench_bh.exe bench_bh.cpp
// Run:   ./bin/bench_bh.exe [size_gib] [passes] [threads_csv]
//        e.g. ./bin/bench_bh.exe 2 15 8,16

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

static float* g_buf = nullptr;
static size_t g_n_floats = 0;

struct ThreadArg {
    size_t begin, end;
    int    mode;
    double sink;
};

static DWORD WINAPI worker(LPVOID p) {
    ThreadArg* a = static_cast<ThreadArg*>(p);
    const float* f = g_buf;
    size_t i = a->begin, e = a->end;

    if (a->mode == 0) {
        // pure read: fold lane sign bits so loads can't be elided
        uint32_t acc = 0;
        for (; i + 8 <= e; i += 8) {
            __m256 v = _mm256_load_ps(f + i);
            acc ^= static_cast<uint32_t>(_mm256_movemask_ps(v));
        }
        a->sink = static_cast<double>(acc);
    } else {
        // MoE-like: stream "weights" and accumulate (memory bound)
        __m256 s    = _mm256_set1_ps(1.0000001f);
        __m256 accv = _mm256_setzero_ps();
        for (; i + 8 <= e; i += 8) {
            __m256 v = _mm256_load_ps(f + i);
            accv = _mm256_fmadd_ps(v, s, accv);
        }
        __m128 lo = _mm256_castps256_ps128(accv);
        __m128 hi = _mm256_extractf128_ps(accv, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        a->sink = static_cast<double>(_mm_cvtss_f32(lo));
    }
    return 0;
}

struct Stats { double min, med, max; };
static Stats summarize(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return { v.front(), v[v.size() / 2], v.back() };
}

int main(int argc, char** argv) {
    double      size_gib   = (argc > 1) ? atof(argv[1]) : 2.0;
    int         passes     = (argc > 2) ? atoi(argv[2]) : 15;
    std::string threads_csv = (argc > 3) ? argv[3] : "8,16";

    SYSTEM_INFO si; GetSystemInfo(&si);
    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms); GlobalMemoryStatusEx(&ms);

    printf("== bench_bh: host DRAM streaming bandwidth (B_H probe) ==\n");
    printf("logical cores: %u | RAM: %.1f GiB total, %.1f GiB avail\n",
           si.dwNumberOfProcessors,
           ms.ullTotalPhys / 1073741824.0, ms.ullAvailPhys / 1073741824.0);

    double avail_gib = ms.ullAvailPhys / 1073741824.0;
    if (size_gib > avail_gib - 2.0) {
        size_gib = std::max(0.5, avail_gib - 2.0);
        printf("(shrunk working set to %.2f GiB for available RAM)\n", size_gib);
    }

    size_t bytes = static_cast<size_t>(size_gib * 1073741824.0);
    g_n_floats = bytes / sizeof(float);

    g_buf = static_cast<float*>(VirtualAlloc(nullptr, bytes,
               MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!g_buf) { fprintf(stderr, "VirtualAlloc failed\n"); return 1; }

    uint32_t rng = 0x9E3779B9u;
    for (size_t i = 0; i < g_n_floats; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        g_buf[i] = static_cast<float>(rng % 2001 - 1000) / 1000.0f;
    }

    LARGE_INTEGER freq; QueryPerformanceFrequency(&freq);

    std::vector<int> tcounts;
    {
        std::string s = threads_csv; size_t pos = 0;
        while (pos <= s.size() && !s.empty()) {
            size_t comma = s.find(',', pos);
            std::string tok = s.substr(pos, comma == std::string::npos ?
                                       std::string::npos : comma - pos);
            if (!tok.empty()) tcounts.push_back(atoi(tok.c_str()));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    std::sort(tcounts.begin(), tcounts.end());
    tcounts.erase(std::unique(tcounts.begin(), tcounts.end()), tcounts.end());

    double sink_total = 0.0;
    printf("\n%-6s %-8s %-10s %-10s %-10s\n", "mode", "threads", "min_GB/s", "med_GB/s", "max_GB/s");

    for (int mode = 0; mode <= 1; ++mode) {
        const char* mname = (mode == 0) ? "read" : "fma";
        for (int nt : tcounts) {
            std::vector<double> gbps;
            std::vector<HANDLE> ths(nt);
            std::vector<ThreadArg> args(nt);
            size_t chunk = (g_n_floats / (size_t)nt) & ~(size_t)7;

            for (int p = -1; p < passes; ++p) {           // p=-1 is warmup
                for (int t = 0; t < nt; ++t) {
                    args[t].begin = chunk * (size_t)t;
                    args[t].end   = (t == nt - 1) ? g_n_floats
                                                  : chunk * (size_t)(t + 1);
                    args[t].mode = mode;
                    args[t].sink = 0.0;
                }
                LARGE_INTEGER t0, t1;
                QueryPerformanceCounter(&t0);
                for (int t = 0; t < nt; ++t)
                    ths[t] = CreateThread(nullptr, 0, worker, &args[t], 0, nullptr);
                WaitForMultipleObjects((DWORD)nt, ths.data(), TRUE, INFINITE);
                QueryPerformanceCounter(&t1);
                for (int t = 0; t < nt; ++t) {
                    CloseHandle(ths[t]);
                    sink_total += args[t].sink;
                }
                if (p >= 0) {
                    double sec = (t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
                    gbps.push_back((double)bytes / sec / 1e9);
                }
            }
            Stats st = summarize(gbps);
            printf("%-6s %-8d %-10.1f %-10.1f %-10.1f\n",
                   mname, nt, st.min, st.med, st.max);
            printf("RESULT mode=%s threads=%d med_gbps=%.1f\n", mname, nt, st.med);
        }
    }
    printf("\n(consumed-checksum %.3f — ignore)\n", sink_total);
    VirtualFree(g_buf, 0, MEM_RELEASE);
    return 0;
}
