#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if(bias != nullptr)CHECK_SAME_DEVICE(out, in, weight, bias);
    else CHECK_SAME_DEVICE(out, in, weight);
    
    // Only support contiguous inputs with correct shape for now.
    ASSERT(in->ndim() == 2, "Linear: input tensor must be 2D.");
    ASSERT(weight->ndim() == 2, "Linear: weight tensor must be 2D.");
    if(bias != nullptr) {
        ASSERT(bias->ndim() == 1, "Linear: bias tensor must be 1D.");
        ASSERT(weight->shape()[0] == bias->shape()[0], "Linear: weight's first dimension must match bias's dimension.");
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(), "Linear: all tensors must be contiguous.");
    } else {
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");
    }
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: input's last dimension must match weight's last dimension.");
    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0], "Linear: output shape is incorrect.");
    
    bool use_bias = (bias != nullptr);

    // always support cpu calculation
    int row_x = static_cast<int>(in->shape()[0]);
    int col_x = static_cast<int>(in->shape()[1]);
    int row_w = static_cast<int>(weight->shape()[0]);
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        if(use_bias)
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), use_bias, row_x,  col_x, row_w);
        else
        return cpu::linear(out->data(), in->data(), weight->data(), nullptr, out->dtype(), use_bias, row_x,  col_x, row_w);
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), use_bias, row_x,  col_x, row_w);
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
