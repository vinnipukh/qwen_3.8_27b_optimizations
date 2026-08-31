Share on Bluesky  Share on Mastadon  Share on LinkedIn  Share on Twitter/X  Share on Reddit  Share on Facebook  Share on Whatsapp  Share via Email

# How to accelerate AI applications on RDNA 3 using WMMA

Originally posted: January 10, 2023

Last updated: February 6, 2024

![Aaryaman Vasishta's avatar](https://gpuopen.com/images/aaryaman-vasishta.CaepVSAE.jpg)

Aaryaman Vasishta

![Takahiro Harada's avatar](https://gpuopen.com/images/takahiro-harada.15XnDK8e.jpg)

Takahiro Harada

Our latest RDNA™ 3 GPUs provide the ability to accelerate Generalized Matrix Multiplication (GEMM) operations. This means that you can now get hardware-accelerated matrix multiplications that take maximum advantage of our new RDNA 3 architecture. This new feature is called **Wave Matrix Multiply Accumulate (WMMA).**

This blog is a quick how-to guide for using the WMMA feature with our RDNA 3 GPU architecture using a _Hello World_ example. It shows how to use the WMMA as a compiler intrinsic in HIP. As a prerequisite, we recommend reading through table 2 in section 1.1 of the [**RDNA 3 ISA guide**](https://developer.amd.com/wp-content/resources/RDNA3_Shader_ISA_December2022.pdf) for an overview of the various terminologies used. It is also recommended to go through the details of how WMMA works in the sections below before jumping straight to the source code examples. As a supplement to this blog, you can also refer to the [**AMD Matrix Instruction Calculator**](https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator) tool to generate in-depth information such as register mappings for every WMMA instruction available.

AMD GPUs based on the RDNA 3 architecture execute WMMA instructions in a very efficient manner allowing applications to achieve excellent performance and utilization. A single WMMA instruction coordinates 32 clocks of optimal work scheduling. AMD’s Mike Mantor, Corporate Fellow and Chief GPU Architect explains it like this:

> **_The WMMA instruction optimizes the scheduling of data movement and peak math operations with minimal VGPR access by providing source data reuse and intermediate destination data forwarding operations without interruption. The regular patterns experienced in matrix operations enable WMMA instructions to reduce the required power while providing optimal operations that enable sustained operations at or very near peak rates._**

WMMA supports inputs of FP16 or BF16 that can be useful for training online or offline, as well as 8-bit and 4-bit integer data types suitable for inference. The table below compares the theoretical FLOPS/clock/CU (floating point operations per clock, per compute unit) of our flagship Radeon RX 7900 XTX GPU based on the RDNA 3 architecture over the previous flagship Radeon RX 6950 XT GPU based on RDNA 2 for different data types:

(IU8 and IU4 refers to the unsigned 8-bit integer datatype and unsigned 4-bit integer datatype respectively)

| **Data type** | **RX 6950 XT FLOPS/clock/CU** | **RX 7900 XTX FLOPS/clock/CU** |
| --- | --- | --- |
| FP16 | 256 | 512 |
| BF16 | N/A | 512 |
| IU8 | 512 | 512 |
| IU4 | 1024 | 1024 |

## WMMA Overview

Unlike traditional per-thread matrix multiplication, WMMA allows the GPU to perform matrix multiplication cooperatively across an entire wavefront of 32 threads in wave32 mode or 64 threads in wave64 mode. This provides the benefit of sharing input/output matrix data across lanes of a wave, thus optimizing the VGPR usage and reducing memory traffic.

Suppose we have the GEMM operation using matrices A, B, C, and D:

`D = A*B + C`

where A and B are the input matrices, C is the accumulator matrix, and D is the destination matrix, also known as the result matrix.

If C isn’t used (e.g. in cases where you don’t use biases in your neural network), you can initialize C to 0 and re-use it as the result matrix:

`C = A*B + C`

This can be illustrated in the figure below, where matrices A, B, C, and D are all using a tile size of 16x16:

![](https://gpuopen.com/images/matrices_illustration.BeVovM6B.png)

As of writing this blog, the three ways you can use WMMA can be via the compiler intrinsic which is available in LLVM clang built-ins, or writing inline assembly on your own, or you can also use rocWMMA, which will allow developers to get access to WMMA-based matrix operations (more details towards the end of this blog). We will focus on the compiler intrinsic approach in this blog.

## How to use WMMA compiler intrinsic

The WMMA compiler intrinsic follows a certain syntax which is described follows:

`D_frag = __builtin_amdgcn_wmma_<C, D format>_16x16x16_<A, B format>_w<32 or 64>(A_frag, B_frag, C_frag, OPSEL)`

If you want to re-use D as C, where C is initialized to zero, simply replace D with C:

`C_frag = __builtin_amdgcn_wmma_<C, D format>_16x16x16_<A, B format>_w<32 or 64>(A_frag, B_frag, C_frag, OPSEL)`

Here, the “C, D format” refers to the format of matrices C and D respectively, which can be any one of f16, f32, or bf16 for floating point datatypes, and i32 as an integer datatype.

The “A,B format” refers to the input matrices A and B respectively, the format of which can be any one of f16, bf16, iu8, or iu4.

The 16x16x16 represents the GEMM convention for the tile size for a MxNxK matrix multiply, where matrix A is of size MxK, matrix B of size KxN, and matrix C/D is of size MxN. In the case of RDNA 3, only 16x16 tile sizes are supported. If your matrix is larger than 16x16, then split it into chunks of 16x16 which can then be passed into the WMMA instruction. In the context of a wave, internally the WMMA instruction takes a tile of 16x16 for matrix A and a tile of 16x16 for matrix B. It then multiplies them to give a 16x16 tile, which is then added with matrix C to give the final 16x16 matrix D tile.

The `w<32 or 64>` in the intrinsic describes whether WMMA is running in wave32 mode or wave64 mode. Depending on the mode, the loading and storing behavior of the matrices may vary. We will describe the differences later in this blog.

The final parameter “OPSEL” will also be explained a little later in this blog. For now, let’s focus on how these matrix fragments (`A_frag`, `B_frag`, `C_frag`, and `D_frag`) are loaded and used.

## Loading elements into matrix fragments

The `A_frag`, `B_frag`, `C_frag`, and `D_frag` parameters are the matrix fragments holding 16 elements each of matrices A, B, C and D respectively. From the perspective of a single lane (thread) within a wave, each “fragment” is locally stored in VGPRs, with each VGPR being 32 bits wide. Each thread holds `A_frag` and `B_frag` in 8 VGPRs for fp16/bf16, 4 VGPRs for iu8, and 2 VGPRs for iu4 regardless of the wave size.

`C_frag` and `D_frag` requires 8 VGPRs in wave32 mode and 4 VGPRs in wave64 mode, regardless of the datatype used by matrices C and D.

It is important to note that WMMA on RDNA 3 requires that the contents of `A_frag` and `B_frag` are replicated between lanes 0-15 and lanes 16-31 of the wave in wave32 mode. This means that for wave32 mode, each VGPR in lane 0 must have the exact same matrix data as each VGPR in lane 16. It is similar for lane 1 into lane 17, and so on, all the way to lane 15 into lane 31. This effectively maintains two copies of matrix data between the two half-waves. In wave64 mode, data from lanes 0-15 must also be replicated into lanes 32-47 and 48-63.

There are currently 12 such WMMA intrinsics following the above syntax. They are broadly divided into two categories, wave32 and wave64, described below:

| **wave32** | **wave64** | **Matrix A,B format** | **Matrix C,D format** |
| --- | --- | --- | --- |
| `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w64` | FP16 | FP32 |
| `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` | `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64` | BF16 | FP32 |
| `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` | `__builtin_amdgcn_wmma_f16_16x16x16_f16_w64` | FP16 | FP16 |
| `__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32` | `__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64` | BF16 | BF16 |
| `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` | `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64` | IU8 | I32 |
| `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` | `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64` | IU4 | I32 |

Finally, the “OPSEL” parameter is a boolean flag required to be specified when using a 16-bit format for the C and D matrices. If this flag is set to true, the elements of C and D are stored in the upper half of the VGPR. However, when this flag is set to false, they are stored in the lower half of the VGPR. If you prefer 0-indexing, set this flag to false. This is illustrated in the code snippet taken from our _Hello World_ example below, where we are storing 16-bit elements from `C_frag` into matrix C:

**OPSEL pseudocode**

```
// call the WMMA intrinsic with OPSEL set to "false"

c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

// 8 VGPRs per C,D fragment per thread in wave32 mode

const int lane = threadIdx.x % 16;

for (int ele = 0; ele < 8; ++ele)

{

    // index into matrix C

    const int r = ele * 2 + (lIdx / 16);

    // store results from unpacked c_frag output

    c[16 * r + lane] = c_frag[ele*2];

}
```

Note the line that stores the matrix C elements from `C_frag` when OPSEL is set to “false”:

**OPSEL=“false”**

```
c[16 * r + lane] = c_frag[ele*2];
```

If OPSEL was set to “true”, then the line above would instead be:

**OPSEL=“true”**

```
c[16 * r + lane] = c_frag[ele*2 + 1];
```

Note that in this particular example, we choose matrix C to always store the `C_frag` elements in packed form, so the OPSEL flag should only affect the indexing in the right-hand side of this expression. You are free to modify this to store in unpacked format based on your app’s requirements.

**WMMA requires a combination of row-major and column-major inputs for matrices A, B, C, and D**. Matrix A is stored in column-major order, whereas matrices B, C, and D are all stored in row-major order. Matrices A and B are stored in packed format (i.e. each VGPR packs 2 fp16 values, 4 iu8 values, or 8 iu4 values), whereas matrices C and D are stored in unpacked format, with the “OPSEL” parameter used to describe the location of storage within the VGPR, described further below.

w32 or w64 represents wave32 or wave64 respectively. It represents the number of threads that will participate in the 16x16x16 GEMM operation.

## Example: \_\_builtin\_amdgcn\_wmma\_f16\_16x16x16\_f16\_w32

Here we will demonstrate how to use `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` to perform a 16x16x16 GEMM with fp16 inputs and outputs in wave32 mode.

The following figure shows the input matrix layout for matrices A and B. For matrix A, each (i, j) in a cell represents the i-th row and j-th column. For matrix B, each (i, j) in a cell represents the i-th column and j-th row.

![](https://gpuopen.com/images/f16_16x16x16_f16_w32_input.rPhA4gZj.png)

From the perspective of a thread, each VGPR holds two packed fp16 elements, with each set of 8 VGPRs holding 16 elements for matrices A and B respectively. Matrix A holds 16 columns in VGPRs whereas matrix B holds 16 rows in VGPRs.

Note the blue, green, and yellow colored cells representing 3 rows of the matrix A and similarly 3 columns of matrix B. These will be mapped to the following figure which shows the layouts of matrices C and D. Also note here the 8 VGPRs per lane would store the elements of C and D in unpacked format, with the 16 bit elements stored in the upper or lower half of the 32-bit VGPR based on the “OPSEL” flag. In our case, OPSEL is set to 0 (False) so each VGPR holds a matrix element in the lower half of the VGPR (bits 0 to 15). Note that, as mentioned before, matrices C and D here are stored in row-major format.

![](https://gpuopen.com/images/f16_16x16x16_f16_w32_output.Rh1wZT11.png)

The following is a code example with some helpful comments, showing how to perform a matrix multiplication of two f16 matrices A and B, and re-using C as D for the GEMM operation C = AB + C in wave32 mode using `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32`

wmma\_test.cpp:

**WMMA example**

```
// Wave Matrix Multiply Accumulate (WMMA) using HIP compiler intrinsic

// Does a matrix multiplication of two 16x16, fp16 matrices, and stores them into a 16x16 fp16 result matrix

#include <iostream>

#include <hip/hip_runtime.h>

#include <hip/hip_fp16.h>

using namespace std;

// Use half16 as an alias of the internal clang vector type of 16 fp16 values

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

__global__ void wmma_matmul(__half* a, __half* b, __half* c)

{

    const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;

    const int lIdx = threadIdx.x;

    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and b

    // a_frag will store one column of the 16x16 matrix A tile

    // b_frag will store one row of the 16x16 matrix B tile

    half16 a_frag;

    half16 b_frag;

    // initialize c fragment to 0

    half16 c_frag = {};

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3

    const int lane = lIdx % 16;

    for (int ele = 0; ele < 16; ++ele)

    {

        b_frag[ele] = b[16*ele + lane];

    }

    for (int ele = 0; ele < 16; ++ele)

    {

        a_frag[ele] = a[16 * lane + ele];

    }

    // call the WMMA intrinsic with OPSEL set to "false"

    c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

    for (int ele = 0; ele < 8; ++ele)

    {

        const int r = ele * 2 + (lIdx / 16);

        // store results from unpacked c_frag output

        c[16 * r + lane] = c_frag[ele*2];

        // if OPSEL was set to "true", the line above would instead be

        // c[16 * r + lane] = c_frag[ele*2 + 1];

    }

}

int main(int argc, char* argv[])

{

    __half a[16 * 16] = {};

    __half b[16 * 16] = {};

    __half c[16 * 16] = {};

    __half *a_gpu, *b_gpu, *c_gpu;

    hipMalloc(&a_gpu, 16*16 * sizeof(__half));

    hipMalloc(&b_gpu, 16*16 * sizeof(__half));

    hipMalloc(&c_gpu, 16*16 * sizeof(__half));

    // fill in some data into matrices A and B

    for (int i = 0; i < 16; ++i)

    {

        for (int j = 0; j < 16; ++j)

        {

            a[i * 16 + j] = (__half)1.f;

            b[i * 16 + j] = (__half)1.f;

        }

    }

    hipMemcpy(a_gpu, a, (16*16) * sizeof(__half), hipMemcpyHostToDevice);

    hipMemcpy(b_gpu, b, (16*16) * sizeof(__half), hipMemcpyHostToDevice);

    hipMemcpy(c_gpu, c, (16*16) * sizeof(__half), hipMemcpyHostToDevice);

    wmma_matmul<<<dim3(1), dim3(32, 1, 1), 0, 0>>>(a_gpu, b_gpu, c_gpu);

    hipMemcpy(c, c_gpu, (16 * 16) * sizeof(__half), hipMemcpyDeviceToHost);

    hipFree(a_gpu);

    hipFree(b_gpu);

    hipFree(c_gpu);

    for (int i = 0; i < 16; ++i)

    {

        for (int j = 0; j < 16; ++j)

        {

            printf("%f ", (float)c[i * 16 + j]);

        }

        printf("\\n");

    }

    return 0;

}
```

The process in the above code is as follows:

1. Initialize the input matrices A and B.

2. Set matrix C to zero and re-use it as matrix D.

3. Pass matrix C to the “wmma\_matmul” kernel, which loads the matrix elements into their respective fragments.

4. Call the WMMA intrinsic.

5. Store the result from c\_frag into matrix C.


To compile the above program on your Radeon RX 7900 XTX or 7900 XT GPU on Linux or Windows using HIP, simply use `hipcc --offload-arch=gfx1100 wmma_test.cpp -o wmma_test`. Make sure you have ROCm v5.4 or newer installed on your Linux environment, or the latest HIP SDK installed on your Windows® environment.

As an alternative to installing the HIP SDK, head to the [Orochi GitHub repository](https://github.com/GPUOpen-LibrariesAndSDKs/Orochi/pull/62) for an example involving usage of hipRTC APIs to compile and run the above code at runtime on Windows® or Linux!

On a side note, if you’re used to using `nvcuda::wmma` APIs and/or rocWMMA, you will notice many similarities here. For example, these matrix fragments in `a_frag`, `b_frag`, `c_frag`, and `d_frag` can be considered the same as the fragment templated type available in those APIs, with the loading and storing of fragments similar to `load_matrix_sync` and `store_matrix_sync` respectively. The call to the compiler intrinsic is similar to calling mma\_sync. The main difference here is that you are doing the loading, storing, syncing, and WMMA calls yourself, rather than relying on the API to do it for you. For brevity, we’ve skipped the synchronization part as it is not needed for a simple example such as the one above, however we do recommend using `__syncthreads()` wherever appropriate.

## WMMA use cases

WMMA can be used to accelerate any use case that involves matrix multiplication. Here we describe three such use cases that are either already available or will be coming soon:

1. Stable diffusion uses WMMA to boost its performance via the SHARK MLIR/IREE runtime for RDNA 3 GPUs

2. AMD’s [Composable Kernels (CK) library](https://github.com/ROCmSoftwarePlatform/composable_kernel) will soon be updated in a new release to support WMMA, which will enable Meta’s AI Template (AIT) library to support end-to-end hardware acceleration for model inference on RDNA 3.

3. The [Machine Intelligence Shader Autogen (MISA)](https://github.com/ROCmSoftwarePlatform/MISA) library will soon release WMMA support to accelerate models like Resnet50 for a performance uplift of roughly 2x from RDNA 2.


## rocWMMA support

So far, we’ve discussed how to use WMMA via compiler intrinsics. However, it may be cumbersome to integrate this with existing CUDA-based applications that utilize the `nvcuda::wmma` APIs via. mma.h (note that the WMMA in `nvcuda::wmma` refers to **Warp** Matrix-Multiply Accumulate, which is different from the **Wave** Matrix-Multiply Accumulate described here).

While it’s true that these intrinsics can be mapped easily to the `mma_sync` API call, the matrix loading/storing and synchronization can be tricky to handle and debug, especially for novice users.

RDNA 3 WMMA support is now available in [**rocWMMA**](https://github.com/ROCmSoftwarePlatform/rocWMMA). This library is portable with `nvcuda::wmma` and it supports MFMA and WMMA instructions, thus allowing your application to have hardware-accelerated ML in both RDNA 3 and CDNA 1/2 based systems.

## References

- [AMD Matrix cores](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)

- [AMD RDNA 3 Instruction Set Architecture Reference Guide](https://developer.amd.com/wp-content/resources/RDNA3_Shader_ISA_December2022.pdf)


## Acknowledgements

Thanks to Atsushi Yoshimura, Joseph Greathouse and Chris Millette for suggesting improvements and providing feedback, and thanks to Mike Mantor for contributing the insightful WMMA explanations.
Matrix layout figures generated by using modified TikZ programs originally written by Damon McDougall.

![Aaryaman Vasishta's avatar](https://gpuopen.com/images/aaryaman-vasishta.CaepVSAE.jpg)

### Aaryaman Vasishta

Aaryaman Vasishta is a researcher and software engineer at AMD’s Advanced Rendering Research group. He completed his master’s from the University of Tokyo, focusing on research in learning proposal kernels for Markov Chain Monte Carlo methods using neural networks. His research interests include real-time ray tracing, real-time neural rendering, and GPGPU.

![Takahiro Harada's avatar](https://gpuopen.com/images/takahiro-harada.15XnDK8e.jpg)

### Takahiro Harada

Takahiro Harada is an engineer working on research and development. His research interests include rendering, ray tracing, and neural networks.

### Related software

[![AMD FidelityFX™ Single Pass Downsampler (SPD)](https://gpuopen.com/images/FFX_SPD.COxnqQ53.jpg)\\
\\
AMD FidelityFX™ Single Pass Downsampler (SPD) \\
\\
AMD FidelityFX Single Pass Downsampler (SPD) provides an AMD RDNA™ architecture optimized solution for generating up to 12 MIP levels of a texture.](https://gpuopen.com/fidelityfx-spd/)[![AMD FidelityFX™ Parallel Sort](https://gpuopen.com/images/fullsize_FFX_ParallelSort_featured.DE93yBTI.jpg)\\
\\
AMD FidelityFX™ Parallel Sort \\
\\
AMD FidelityFX Parallel Sort makes sorting data on the GPU quicker, and easier. Use our SM6.0 compute shaders to get your data in order.](https://gpuopen.com/fidelityfx-parallel-sort/)[![HIP Ray Tracing](https://gpuopen.com/images/featured-231825263-A_AMD_HIP_Ray_Tracing_Lockup_RGB_Wht.DN2Oy7i-.jpg)\\
\\
HIP Ray Tracing \\
\\
HIP RT is a ray tracing library for HIP, making it easy to write ray tracing applications in HIP.](https://gpuopen.com/hiprt/)[![Orochi](https://gpuopen.com/images/featured-231825264-A_AMD_Orochi_Lockup_RGB_Wht.YcgioIhl.jpg)\\
\\
Orochi \\
\\
Orochi is a library which loads HIP and CUDA® APIs dynamically, allowing the user to switch APIs at runtime.](https://gpuopen.com/orochi/)

### Related news and technical articles

[![MiniDXNN v0.4.0: Interactive neural texture compression on DirectX 12](https://gpuopen.com/images/minidxnn04_featured.D2BFPCTo.jpg)\\
\\
MiniDXNN v0.4.0: Interactive neural texture compression on DirectX 12 \\
\\
MiniDXNN v0.4.0 introduces D3D12 Linear Algebra (SM 6.10) support, input encodings and neural texture compression, plus a real-time GUI app that trains and visualizes GPU-accelerated MLPs on DirectX® 12.](https://gpuopen.com/learn/minidxnn-v040-interactive-neural-texture-compression/)[![GPU view-adaptive crack-free subdivision of Bézier surfaces](https://gpuopen.com/images/teaser_subdiv_wire_featured.BtvGrp40.jpg)\\
\\
GPU view-adaptive crack-free subdivision of Bézier surfaces \\
\\
Learn how fast, crack-free GPU work graph subdivision for bicubic Bézier surfaces dramatically reduce triangle counts while simplifying implementation and matching hardware-tessellation quality.](https://gpuopen.com/learn/gpu-view-adaptive-subdivision/)[![Ray tracing massive amounts of animated geometry using tetrahedral cages](https://gpuopen.com/images/rt_massive_teaser_gamma_featured.C4yHI1jh.jpg)\\
\\
Ray tracing massive amounts of animated geometry using tetrahedral cages \\
\\
Animate compact tetrahedral cages and reuse static mini-BLASes to ray-trace hundreds of millions of triangles in real time, dramatically cutting per-frame update and memory costs for dense foliage, grass, and crowds.](https://gpuopen.com/learn/ray-tracing-massive-amounts-animated-geometry/)[![Post-mortem GPU crash debugging with LLMs](https://gpuopen.com/images/gpu_crash_debug_llm_featured.HFp4Dy93.jpg)\\
\\
Post-mortem GPU crash debugging with LLMs \\
\\
The new AMD RGD MCP Server connects LLM agents to AMD's GPU crash analysis pipeline, turning a single prompt into root-cause analysis and source-code fix suggestions.](https://gpuopen.com/learn/post-mortem-gpu-crash-debugging-with-llms/)[![AMD FSR Upscaling 4.1 RDNA 3 Support Now Available in FSR SDK 2.3 Update](https://gpuopen.com/images/featured_white_amd_fsr_sdk.CSz0w4hi.png)\\
\\
AMD FSR Upscaling 4.1 RDNA 3 Support Now Available in FSR SDK 2.3 Update \\
\\
AMD FSR "Redstone" SDK 2.3 brings ML-powered FSR Upscaling 4.1.1 to AMD Radeon RX 7000 Series GPUs, along with Frame Generation 4.0.1 and Ray Regeneration 1.2 improvements for RDNA 4 GPUs.](https://gpuopen.com/learn/amd-fsr-sdk-2-3-blog/)[![New AMD Radeon Developer Tool Suite update brings shader source code, Extended PIX Markers, and command-line capture](https://gpuopen.com/images/rdts_pix_marker_update_featured.BwSPmF-Q.jpg)\\
\\
New AMD Radeon Developer Tool Suite update brings shader source code, Extended PIX Markers, and command-line capture \\
\\
The new AMD Radeon Developer Tool Suite release delivers RGP 2.7 with shader source code viewing, instruction‑level divergence metrics, and Extended PIX Marker support, expanded hardware compatibility, and updates across RGD, RRA, RMV, RGA, and RDP.](https://gpuopen.com/learn/radeon-developer-tool-suite-shader-source-code/)[![WMMA guide for AMD RDNA 4 architecture GPUs - part 3](https://gpuopen.com/images/wmma_rdna4_pt3_featured.ClZjn8Mn.jpg)\\
\\
WMMA guide for AMD RDNA 4 architecture GPUs - part 3 \\
\\
Learn how to implement fast in-register matrix transpose on AMD RDNA™ 4 architecture GPUs with a WMMA-based identity trick, delivering a lightweight, memory-free alternative proven in Llama.cpp.](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-3/)[![WMMA guide for AMD RDNA 4 architecture GPUs - part 2](https://gpuopen.com/images/wmma_rdna4_pt2b_featured.CdSF_7ru.jpg)\\
\\
WMMA guide for AMD RDNA 4 architecture GPUs - part 2 \\
\\
Achieve peak AMD RDNA™ 4 architecture memory bandwidth for low-precision GEMM by fusing WMMA to double the K dimension, enabling 128-bit loads for FP8/INT8, and matching hipBLAS results bit-for-bit.](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/)

### Related videos

[![Advancing AI in video games with AMD Schola | HTEC Days 2025 - YouTube link](https://gpuopen.com/images/Schola_HTEC_video_thumb.wpRznDZ7.jpg)\\
\\
Advancing AI in video games with AMD Schola \| HTEC Days 2025 - YouTube link \\
\\
Join Alexander Cann, Lead Developer at Schola, and Mehdi Saeedi, AI Lead at Schola, as they take you through the fascinating world of reinforcement learning (RL) and its transformative impact on gaming. They'll be joined by Gabor Sines, Sr. Fellow Engineer at AMD, as moderator.](https://gpuopen.com/videos/amd-schola-htec-days-2025/)[![Two-level radiance caching for fast and scalable real-time dynamic GI in games (GDC 2023 - YouTube link)](https://gpuopen.com/images/AGS-GI-Social.C4Ee7YdG.jpg)\\
\\
Two-level radiance caching for fast and scalable real-time dynamic GI in games (GDC 2023 - YouTube link) \\
\\
This presentation is a practical implementation of a solution aimed at making the most of every sample by caching the estimated radiance into a cache hierarchy used for both sampling and filtering.](https://gpuopen.com/videos/two-level-radiance-caching-gi-gdc-2023/)[![Optimizing Game Performance with the Radeon Developer Tool Suite (GDC 2023 - YouTube link)](https://gpuopen.com/images/Optimizing-Game-Performance-with-the-Radeon-Developer-Tool-Suite-Thumbnail-featured-1.CLTju8MW.jpg)\\
\\
Optimizing Game Performance with the Radeon Developer Tool Suite (GDC 2023 - YouTube link) \\
\\
This talk gives an overview of RGP, RMV, RRA, and RGA, introducing new features and improvements, and reveal the current work in progress.](https://gpuopen.com/videos/gdc-2023-optimizing-game-performance-with-rdts/)[![AMD Ryzen™ Processor Software Optimization (GDC 2023 - YouTube link)](https://gpuopen.com/images/AMD-Ryzen-Processor-Software-Optimization-Thumbnail-featured.BLO83WQh.jpg)\\
\\
AMD Ryzen™ Processor Software Optimization (GDC 2023 - YouTube link) \\
\\
In this talk from learn about AMD Ryzen™ products featuring advanced technologies, including laptop, desktop, and workstation processors.](https://gpuopen.com/videos/gdc-2023-amd-ryzen-processor-software-optimization/)[![Real-time Sparse Distance Fields for Games (GDC 2023 - YouTube link)](https://gpuopen.com/images/Real-time-Sparse-Distance-Fields-Thumbnail-featured.5npMQq-W.jpg)\\
\\
Real-time Sparse Distance Fields for Games (GDC 2023 - YouTube link) \\
\\
This presentation introduces a novel algorithm for PC and console developers to efficiently generate sparse distance fields in real-time.](https://gpuopen.com/videos/gdc-2023-real-time-sparse-distance-fields-for-games/)[![Game Optimization: Radeon™ Developer Tools on RADV and Steam Deck™ (Vulkanised 2023 - YouTube link)](https://gpuopen.com/images/Vulkanised_2023_Talk_featured.xnnF4a8q.jpg)\\
\\
Game Optimization: Radeon™ Developer Tools on RADV and Steam Deck™ (Vulkanised 2023 - YouTube link) \\
\\
This talk at Vulkanised 2023 covers how to use the Radeon Developer Tool Suite (RDTS) to optimize games using RADV and Steam Deck.](https://gpuopen.com/videos/radeon-developer-tools-radv-steam-deck/)[![Compute Shaders - Game Industry Conference 2021](https://gpuopen.com/images/Compute_Shaders_title_page.BnwShBuy.png)\\
\\
Compute Shaders - Game Industry Conference 2021 \\
\\
This talk introduces compute shaders, explaining ideas from a software and hardware perspective, as well as considerations when writing compute shaders.](https://gpuopen.com/videos/compute-shaders-gic21/)[![AMD Ryzen™ Processor Software Optimization (GDC 2022) - YouTube link](https://gpuopen.com/images/AMD-Ryzen%E2%84%A2-Processor-Software-Optimization-thumbnail.Cxn8MGh3.png)\\
\\
AMD Ryzen™ Processor Software Optimization (GDC 2022) - YouTube link \\
\\
Join AMD for an introduction to the AMD Ryzen™ family of processors which power today’s game consoles and PCs.](https://gpuopen.com/videos/ryzen-software-optimization/)