#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *in, std::byte *weight, float eps, llaisysDataType_t type, int in_rows, int in_cols);
}
