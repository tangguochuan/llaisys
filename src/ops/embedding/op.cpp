#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    // index must be Int64
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must be Int64 type.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");
    
    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->numel(), weight->shape().back());
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->numel(), weight->shape().back());
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
