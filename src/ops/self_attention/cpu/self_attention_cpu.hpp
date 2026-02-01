#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, std::byte *q, std::byte *k, std::byte *v, float scale, llaisysDataType_t type,
                    int seqlen, int nhead, int d, int total_len, int nkvhead, int dv);
}
