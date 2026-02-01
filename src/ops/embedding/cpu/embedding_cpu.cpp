#include "embedding_cpu.hpp"

#include "../../../utils.hpp"
#include <cstring>
template <typename T>
void embedding_(T *out, std::byte *index, T *weight, int stride,int index_numel) {
    for(int i = 0; i < index_numel; i++){
        int64_t idx = static_cast<int64_t>(reinterpret_cast<int64_t*>(index)[i]);
        T* src = weight + idx * stride;
        T* dst = out + i * stride;
        std::memcpy(dst, src, stride * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t type, size_t index_numel, size_t weight_stride) {
    switch (type)
    {
    case LLAISYS_DTYPE_F32:
        {
           return embedding_(reinterpret_cast<float *>(out), index, reinterpret_cast<float *>(weight), weight_stride, index_numel);            
        }
    case LLAISYS_DTYPE_BF16:
        {
           return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), index, reinterpret_cast<llaisys::bf16_t *>(weight), weight_stride, index_numel);            
        }
    case LLAISYS_DTYPE_F16:
        {
            return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), index, reinterpret_cast<llaisys::fp16_t *>(weight), weight_stride, index_numel);            
        }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
