#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, T *in, T *weight, T *bias, bool use_bias, int row_x, int col_x, int row_w){
    for(int i = 0; i < row_x; i++){
        for(int j = 0; j < row_w; j++){
            float sum = 0;
            for(int k = 0; k < col_x; k++){
                sum = sum + llaisys::utils::cast<float>(in[i * col_x + k]) * llaisys::utils::cast<float>(weight[j * col_x + k]);
            }
            out[i * row_w + j] = llaisys::utils::cast<T>(sum + (use_bias ? llaisys::utils::cast<float>(bias[j]) : 0.0f));
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, llaisysDataType_t type, bool use_bias, int row_x, int col_x, int row_w){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in), reinterpret_cast<float *>(weight), reinterpret_cast<float *>(bias), use_bias, row_x, col_x, row_w);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in), reinterpret_cast<llaisys::bf16_t *>(weight), reinterpret_cast<llaisys::bf16_t *>(bias), use_bias, row_x, col_x, row_w);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in), reinterpret_cast<llaisys::fp16_t *>(weight), reinterpret_cast<llaisys::fp16_t *>(bias), use_bias, row_x, col_x, row_w);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    
}
} // namespace llaisys::ops::cpu
