#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, llaisysDataType_t type, bool use_bias, int row_x, int col_x, int row_w);
}
