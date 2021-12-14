#pragma once

#include "types.h"
#include <motionlib/vectorops.h>

#define BUFFER_SIZE (2048*8)
extern motion_dtype alloc_buffer[BUFFER_SIZE];
extern int alloc_idx;
static inline vptr fast_alloc_vec(size_t sz) {
    if (alloc_idx + sz >= BUFFER_SIZE) {
        alloc_idx = sz;
        return alloc_buffer;
    }
    else {
        vptr ret = alloc_buffer + alloc_idx;
        alloc_idx += sz;
        return ret;
    }
}
