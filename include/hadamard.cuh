// hadamard.cuh — 快速 Hadamard 变换 CUDA kernel 的 host 侧接口
//
// 语义：
//   输入张量逻辑形状 [batch_size, seq_len, num_heads, head_dim]，
//   展平为 [total_tokens, head_dim]，total_tokens = batch_size*seq_len*num_heads。
//   每个 token 独立地对最后一维做 Hadamard 变换：
//     非归一化: y = H x
//     归一化:   y = H x / sqrt(head_dim)        (默认，正交旋转，误差阈值友好)
//
// 设计时预留的扩展位（Week3+ 融合量化 / Tensor Core）：
//   - dtype / normalize 已经是运行时参数；
//   - 后续可在此接口旁新增 launch_hadamard_fused_quant(..., QuantType quant_type, ...)
//     复用同一套 (dtype, head_dim) 分发，而不必改动本接口。
#pragma once

#include <cuda_runtime.h>

// 支持的数据类型。 Week1/Week2 仅要求 FP16 / BF16。
enum class DataType {
  FP16,
  BF16,
};

// host 侧统一入口。
//
// 参数：
//   input/output  设备指针，元素类型为 dtype 对应的 __half / __nv_bfloat16
//   head_dim      2 的幂；当前实例化支持 32/64/128/256/512/1024（64/128/256 为核心）
//   normalize     true: y = Hx/sqrt(head_dim)；false: y = Hx
//   stream        调用方提供的 CUDA stream
//
// 返回值：
//   0  成功（kernel 已入队，不代表已执行完）
//   -1 不支持的 head_dim
//   -2 不支持的 dtype
//
// 注意：CUDA 调用错误在本函数内部用 CUDA_CHECK 直接报错退出，不隐藏错误。
int launch_hadamard(const void* input,
                    void* output,
                    int batch_size,
                    int seq_len,
                    int num_heads,
                    int head_dim,
                    DataType dtype,
                    bool normalize,
                    cudaStream_t stream);

// 工具：数据类型名称（日志/CSV 用）
const char* dtype_name(DataType dtype);
