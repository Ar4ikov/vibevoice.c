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
    -1.0f, -0.6961928f, -0.5250730f, -0.3949338f,
    -0.2844871f, -0.1848489f, -0.0911179f,  0.0f,
     0.0796009f,  0.1609302f,  0.2461123f,  0.3379930f,
     0.4407233f,  0.5626170f,  0.7229568f,  1.0f
};
