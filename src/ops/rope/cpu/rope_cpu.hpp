#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, std::byte *in, std::byte *pos_ids, float theta, llaisysDataType_t type, int seq_len, int nheads, int head_dim);
}
