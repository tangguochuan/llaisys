#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), 
           "SelfAttention: all tensors must be contiguous.");
    
    // Check dimensions
    // q: [seqlen, nhead, d]
    // k: [total_len, nkvhead, d]
    // v: [total_len, nkvhead, dv]
    // attn_val: [seqlen, nhead, dv]
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3D [seqlen, nhead, d].");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3D [total_len, nkvhead, d].");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3D [total_len, nkvhead, dv].");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3D [seqlen, nhead, dv].");
    
    int seqlen = static_cast<int>(q->shape()[0]);
    int nhead = static_cast<int>(q->shape()[1]);
    int d = static_cast<int>(q->shape()[2]);
    
    int total_len = static_cast<int>(k->shape()[0]);
    int nkvhead = static_cast<int>(k->shape()[1]);
    int d_k = static_cast<int>(k->shape()[2]);
    
    int total_len_v = static_cast<int>(v->shape()[0]);
    int nkvhead_v = static_cast<int>(v->shape()[1]);
    int dv = static_cast<int>(v->shape()[2]);
    
    // Validate dimensions
    ASSERT(d == d_k, "SelfAttention: q and k must have the same head dimension d.");
    ASSERT(total_len == total_len_v, "SelfAttention: k and v must have the same total_len.");
    ASSERT(nkvhead == nkvhead_v, "SelfAttention: k and v must have the same nkvhead.");
    ASSERT(nhead % nkvhead == 0, "SelfAttention: nhead must be divisible by nkvhead for GQA.");
    
    ASSERT(attn_val->shape()[0] == static_cast<size_t>(seqlen), "SelfAttention: attn_val shape[0] must match seqlen.");
    ASSERT(attn_val->shape()[1] == static_cast<size_t>(nhead), "SelfAttention: attn_val shape[1] must match nhead.");
    ASSERT(attn_val->shape()[2] == static_cast<size_t>(dv), "SelfAttention: attn_val shape[2] must match dv.");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                   seqlen, nhead, d, total_len, nkvhead, dv);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                   seqlen, nhead, d, total_len, nkvhead, dv);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
