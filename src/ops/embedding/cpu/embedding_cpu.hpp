#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t type, size_t index_numel, size_t weight_stride);
}
