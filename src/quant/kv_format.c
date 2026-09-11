/**
 * @file kv_format.c
 * @brief Names and sizes for the KV-cache storage formats.
 */

#include "vibevoice/kv_quant.h"

#include <string.h>

static const struct {
    vv_kv_format_t fmt;
    const char*    name;
    const char*    alias;
    int            bits_x100;
} k_formats[] = {
    { VV_KV_FP32,     "fp32",     "f32",      3200 },
    { VV_KV_FP16,     "fp16",     "f16",      1600 },
    { VV_KV_FP8_E4M3, "fp8",      "fp8-e4m3",  800 },
    { VV_KV_FP8_E5M2, "fp8-e5m2", "e5m2",      800 },
    { VV_KV_TQ4,      "tq4",      "turbo4",    400 },
    { VV_KV_TQ3,      "tq3",      "turbo3",    300 },
    { VV_KV_TQ2,      "tq2",      "turbo2",    200 },
    { VV_KV_TQ1_5,    "tq1.5",    "turbo1.5",  150 },
};

vv_kv_format_t vv_kv_format_parse(const char* name) {
    if (!name || !*name) return VV_KV_FP16;
    for (size_t i = 0; i < sizeof(k_formats) / sizeof(k_formats[0]); i++) {
        if (strcmp(name, k_formats[i].name) == 0 ||
            strcmp(name, k_formats[i].alias) == 0) {
            return k_formats[i].fmt;
        }
    }
    return VV_KV_FORMAT_COUNT;   /* caller reports the bad value */
}

const char* vv_kv_format_name(vv_kv_format_t fmt) {
    for (size_t i = 0; i < sizeof(k_formats) / sizeof(k_formats[0]); i++)
        if (k_formats[i].fmt == fmt) return k_formats[i].name;
    return "?";
}

int vv_kv_format_bits_x100(vv_kv_format_t fmt) {
    for (size_t i = 0; i < sizeof(k_formats) / sizeof(k_formats[0]); i++)
        if (k_formats[i].fmt == fmt) return k_formats[i].bits_x100;
    return 1600;
}
