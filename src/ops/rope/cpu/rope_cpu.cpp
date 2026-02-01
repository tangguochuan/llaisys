#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, T *in, int64_t *pos_ids, float theta, int seq_len, int nheads, int head_dim){
    int half_dim = head_dim / 2;

    for(int i = 0; i < seq_len; i++){
        float pos_id = static_cast<float>(pos_ids[i]);
        
        for(int d = 0; d < half_dim; d++){
            double freq = std::pow(static_cast<double>(theta), -2.0 * d / head_dim);
            double angle = static_cast<double>(pos_id) * freq;
            float s = static_cast<float>(std::sin(angle));
            float c = static_cast<float>(std::cos(angle));
            for(int h = 0; h < nheads; h++){
                int base_idx = i * nheads * head_dim + h * head_dim;
                int idx_a = base_idx + d;
                int idx_b = base_idx + d + half_dim;

                float val_a = llaisys::utils::cast<float>(in[idx_a]);
                float val_b = llaisys::utils::cast<float>(in[idx_b]);

                out[idx_a] = llaisys::utils::cast<T>(val_a * c - val_b * s);
                out[idx_b] = llaisys::utils::cast<T>(val_b * c + val_a * s);
            }
        }
    }
}
namespace llaisys::ops::cpu {
void rope(std::byte *out, std::byte *in, std::byte *pos_ids, float theta, llaisysDataType_t type, int seq_len, int nheads, int head_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in), reinterpret_cast<int64_t *>(pos_ids), theta, seq_len, nheads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in), reinterpret_cast<int64_t *>(pos_ids), theta, seq_len, nheads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in), reinterpret_cast<int64_t *>(pos_ids), theta, seq_len, nheads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    
}
} // namespace llaisys::ops::cpu
