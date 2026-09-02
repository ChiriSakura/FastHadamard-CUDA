// 实现策略：
//   - 输入逻辑展平为 [total_tokens, head_dim]；
//   - 每个 block 处理 1 个 token，blockDim.x = HEAD_DIM（一个线程一个元素，映射最直观）；
//   - 数据先从 global memory 读入并按元素转成 FP32，存放在 shared memory；
//   - 在 shared memory 上做 log2(HEAD_DIM) 轮蝶形操作（快速 Walsh-Hadamard），
//     全部使用 FP32 累加 —— BF16 尤其依赖这一点才能满足误差要求；
//   - 最后按 NORMALIZE 决定是否乘 1/sqrt(HEAD_DIM)，转回原始类型写回。
//
// 蝶形说明：
//   对 stride = 1, 2, 4, ..., HEAD_DIM/2，
//   对每对下标 (i, i^stride)（i 的第 log2(stride) 位为 0）执行
//       (a, b) -> (a + b, a - b)
//   等价于乘以 Sylvester 型 Hadamard 矩阵 H（H @ H^T = HEAD_DIM * I）。
#include "hadamard.cuh"
#include "cuda_check.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>

// ---------------------------------------------------------------------------
// kernel 计算逻辑
// ---------------------------------------------------------------------------

template <typename T, int HEAD_DIM, bool NORMALIZE>
__global__ void hadamard_kernel(const T* __restrict__ input,
                                T* __restrict__ output,
                                int total_tokens, // 逻辑上 [total_tokens, head_dim]
                                float norm_scale) {
  // 确定当前 block 处理的 token 和当前线程处理的元素，每个block处理一个token
  const int token = blockIdx.x;
  if (token >= total_tokens) {
    return;
  }

  // 每个线程处理一个元素，tid = [0, HEAD_DIM)
  const int tid = threadIdx.x;
  // 计算当前 token 在 input/output 中的起始下标
  const size_t base = static_cast<size_t>(token) * HEAD_DIM;

  // 分配 shared memory 并将 input 转为 FP32 存入
  __shared__ float smem[HEAD_DIM];
  smem[tid] = static_cast<float>(input[base + tid]);
  __syncthreads();

  // log2(HEAD_DIM) 轮蝶形操作，每轮 stride = 1, 2, 4, ..., HEAD_DIM/2
#pragma unroll
  for (int stride = 1; stride < HEAD_DIM; stride <<= 1) {
    const int partner = tid ^ stride;

    // 每对元素只由下标较小的线程处理一次
    if (partner > tid) {
      const float a = smem[tid];
      const float b = smem[partner];
      smem[tid] = a + b;
      smem[partner] = a - b;
    }
    __syncthreads();
  }

  // 按需归一化并写回原始类型
  const float scale = NORMALIZE ? norm_scale : 1.0f;
  output[base + tid] = static_cast<T>(smem[tid] * scale);
}

// ---------------------------------------------------------------------------
// 分发辅助（Host-side helper）
// ---------------------------------------------------------------------------

namespace {

// 按 HEAD_DIM 实例化模板并启动。norm_scale 在 host 侧用双精度算好再取 float，
// 避免在 device 端引入不必要的近似。
template <typename T, int HEAD_DIM>
void launch_dim(const void* input, void* output, int total_tokens,
                bool normalize, float norm_scale, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(total_tokens));
  const dim3 block(HEAD_DIM);
  const T* in = static_cast<const T*>(input);
  T* out = static_cast<T*>(output);
  if (normalize) {
    hadamard_kernel<T, HEAD_DIM, true>
        <<<grid, block, 0, stream>>>(in, out, total_tokens, norm_scale);
  } else {
    hadamard_kernel<T, HEAD_DIM, false>
        <<<grid, block, 0, stream>>>(in, out, total_tokens, norm_scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
int launch_typed(const void* input, void* output, int total_tokens,
                 int head_dim, bool normalize, float norm_scale,
                 cudaStream_t stream) {
  switch (head_dim) {
    case 32:
      launch_dim<T, 32>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 64:
      launch_dim<T, 64>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 128:
      launch_dim<T, 128>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 256:
      launch_dim<T, 256>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 512:
      launch_dim<T, 512>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 1024:
      launch_dim<T, 1024>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    default:
      return -1;  // 不支持的 head_dim
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// 公开接口，主要用于对外暴露的 C++ API
// ---------------------------------------------------------------------------

int launch_hadamard(const void* input, void* output, int batch_size,
                    int seq_len, int num_heads, int head_dim, DataType dtype,
                    bool normalize, cudaStream_t stream) {
  const long long total_tokens =
      static_cast<long long>(batch_size) * seq_len * num_heads;
  if (total_tokens == 0) {
    return 0;  // 空输入，直接成功返回
  }
  // host 侧计算归一化系数：1 / sqrt(head_dim)
  const float norm_scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));

  switch (dtype) {
    case DataType::FP16:
      return launch_typed<__half>(input, output, static_cast<int>(total_tokens),
                                  head_dim, normalize, norm_scale, stream);
    case DataType::BF16:
      return launch_typed<__nv_bfloat16>(input, output,
                                         static_cast<int>(total_tokens),
                                         head_dim, normalize, norm_scale, stream);
    default:
      return -2;  // 不支持的 dtype
  }
}

const char* dtype_name(DataType dtype) {
  switch (dtype) {
    case DataType::FP16:
      return "fp16";
    case DataType::BF16:
      return "bf16";
  }
  return "unknown";
}
