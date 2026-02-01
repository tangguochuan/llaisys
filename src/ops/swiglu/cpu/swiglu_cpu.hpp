#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, std::byte *gate, std::byte *up, llaisysDataType_t type, size_t numel);
}
