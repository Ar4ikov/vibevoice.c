/**
 * @file nf4_table.c
 * @brief NF4 (NormalFloat4) lookup table
 *
 * The NF4 quantization format uses 16 fixed values derived from
 * the normal distribution. Each 4-bit index maps to one of these
 * values, which is then scaled by a per-block scale factor.
 */
#include "vibevoice/vibevoice.h"

const float VV_NF4_TABLE[16] = {
    -1.0f,              -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f,  0.0f,
     0.07958029955625534f,  0.16093020141124725f,  0.24611230194568634f,  0.33791524171829224f,
     0.44070982933044434f,  0.5626170039176941f,   0.7229568362236023f,   1.0f
};
