#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, T *q, T *k, T *v, float scale,
                     int seqlen, int nhead, int d, int total_len, int nkvhead, int dv) {
    int num_rep = nhead / nkvhead;
    int kv_offset = total_len - seqlen;

    #pragma omp parallel for collapse(2)
    for (int seq_idx = 0; seq_idx < seqlen; seq_idx++) {
        for (int head_idx = 0; head_idx < nhead; head_idx++) {
            std::vector<float> local_scores(total_len);
            
            int kv_head_idx = head_idx / num_rep;
            T *q_vec = q + (seq_idx * nhead + head_idx) * d;
            int max_kv_pos = seq_idx + kv_offset;
            
            float max_score = -std::numeric_limits<float>::infinity();

            for (int kv_pos = 0; kv_pos <= max_kv_pos; kv_pos++) {
                T *k_vec = k + (kv_pos * nkvhead + kv_head_idx) * d;
                float sum = 0.0f;
                for (int i = 0; i < d; i++) {
                    sum += llaisys::utils::cast<float>(q_vec[i]) * llaisys::utils::cast<float>(k_vec[i]);
                }
                float score = sum * scale;
                local_scores[kv_pos] = score;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            for (int kv_pos = 0; kv_pos <= max_kv_pos; kv_pos++) {
                local_scores[kv_pos] = std::exp(local_scores[kv_pos] - max_score);
                sum_exp += local_scores[kv_pos];
            }
            float inv_sum_exp = 1.0f / (sum_exp + 1e-9f);

            T *out_vec = attn_val + (seq_idx * nhead + head_idx) * dv;
            for (int i = 0; i < dv; i++) {
                float res = 0.0f; 
                for (int kv_pos = 0; kv_pos <= max_kv_pos; kv_pos++) {
                    float w = local_scores[kv_pos] * inv_sum_exp;
                    T *v_vec = v + (kv_pos * nkvhead + kv_head_idx) * dv;
                    res += w * llaisys::utils::cast<float>(v_vec[i]);
                }
                out_vec[i] = llaisys::utils::cast<T>(res);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, std::byte *q, std::byte *k, std::byte *v, float scale, llaisysDataType_t type,
                    int seqlen, int nhead, int d, int total_len, int nkvhead, int dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<float *>(q),
                               reinterpret_cast<float *>(k), reinterpret_cast<float *>(v), scale,
                               seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<llaisys::bf16_t *>(q),
                               reinterpret_cast<llaisys::bf16_t *>(k), reinterpret_cast<llaisys::bf16_t *>(v), scale,
                               seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<llaisys::fp16_t *>(q),
                               reinterpret_cast<llaisys::fp16_t *>(k), reinterpret_cast<llaisys::fp16_t *>(v), scale,
                               seqlen, nhead, d, total_len, nkvhead, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
