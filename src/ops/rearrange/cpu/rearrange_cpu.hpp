#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, std::byte *in, llaisysDataType_t type);
}
