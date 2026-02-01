#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void argmax_(size_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) {
        return;
    }
    *max_idx = 0;
    *max_val = vals[0];
    for (size_t i = 1; i < numel; i++) {
        if(llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(*max_val)) {
            *max_val = vals[i];
            *max_idx = i;
        }
        // if (vals[i] > *max_val) {
        //     *max_val = vals[i];
        //     *max_idx = i;
        // }
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type)
    {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<size_t*>(max_idx), reinterpret_cast<float*>(max_val), reinterpret_cast<float*>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<size_t*>(max_idx), reinterpret_cast<llaisys::bf16_t*>(max_val), reinterpret_cast<llaisys::bf16_t*>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<size_t*>(max_idx), reinterpret_cast<llaisys::fp16_t*>(max_val), reinterpret_cast<llaisys::fp16_t*>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
