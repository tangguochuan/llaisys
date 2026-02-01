#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, T *in, T *weight, float eps, int in_rows, int in_cols){
    for(int i = 0; i < in_rows; i++){
        //计算均方根
        float rms = 0.0f;
        for(int j = 0; j < in_cols; j++){
            float val = llaisys::utils::cast<float>(in[i * in_cols + j]);
            rms += val * val;
        }
        rms = std::sqrt(rms / in_cols + eps);
        for(int j = 0; j < in_cols; j++){
            out[i * in_cols + j] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in[i * in_cols + j])  * llaisys::utils::cast<float>(weight[j]) / rms);
        }
    }
}
namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *in, std::byte *weight, float eps, llaisysDataType_t type, int in_rows, int in_cols) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in), reinterpret_cast<float *>(weight), eps, in_rows, in_cols);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in), reinterpret_cast<llaisys::bf16_t *>(weight), eps, in_rows, in_cols);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in), reinterpret_cast<llaisys::fp16_t *>(weight), eps, in_rows, in_cols);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
