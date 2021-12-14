#include "fast_alloc.h"

#include "types.h"
#include <motionlib/vectorops.h>

motion_dtype alloc_buffer[BUFFER_SIZE];
int alloc_idx = 0;
