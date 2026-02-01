#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    // Only support contiguous inputs with same shape for now.
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: all tensors must be contiguous.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(in->ndim() == 2, "RMSNorm: input tensor must be 2D.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight tensor must be 1D.");
    ASSERT(in->shape()[1] == weight->shape()[0], "RMSNorm: input's last dimension must match weight's dimension.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    int in_rows = static_cast<int>(in->shape()[0]);
    int in_cols = static_cast<int>(in->shape()[1]);

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), in_rows, in_cols);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), in_rows, in_cols);
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
