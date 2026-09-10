/**
 * @file dequant_cpu.c
 * @brief CPU reference implementation of NF4 dequantization
 *
 * Unpacks 4-bit NF4 values from packed uint8 storage, looks up each
 * nibble in the NF4 table, and multiplies by the per-block scale.
 * This is the reference (slow) implementation for validation.
 */
#include "vibevoice/vibevoice.h"

#include <stdint.h>

/** @brief NF4 lookup table (defined in nf4_table.c) */
extern const float VV_NF4_TABLE[16];

vv_status_t vv_dequant_nf4_cpu(const uint8_t* packed,
                                const float*   scales,
                                float*         output,
                                int            n_elements,
                                int            block_size)
{
    if (!packed || !scales || !output) {
        return VV_ERR_NULL_PTR;
    }
    if (n_elements <= 0 || block_size <= 0) {
        return VV_ERR_INVALID_ARG;
    }

    /*
     * Each byte in 'packed' contains two NF4 values:
     *   high nibble = first element  (bits 7..4)
     *   low  nibble = second element (bits 3..0)
     *
     * bitsandbytes convention: byte = (quant[even] << 4) | quant[odd]
     *
     * Elements are grouped into blocks of 'block_size'.
     * Each block has one scale factor in 'scales'.
     */
    int n_packed = (n_elements + 1) / 2;  /* ceil(n_elements / 2) */
    int elem_idx = 0;

    for (int i = 0; i < n_packed && elem_idx < n_elements; i++) {
        uint8_t byte = packed[i];

        /* High nibble (first element) */
        {
            uint8_t hi = (byte >> 4) & 0x0F;
            int block_idx = elem_idx / block_size;
            float scale = scales[block_idx];
            output[elem_idx] = VV_NF4_TABLE[hi] * scale;
            elem_idx++;
        }

        /* Low nibble (second element) */
        if (elem_idx < n_elements) {
            uint8_t lo = byte & 0x0F;
            int block_idx = elem_idx / block_size;
            float scale = scales[block_idx];
            output[elem_idx] = VV_NF4_TABLE[lo] * scale;
            elem_idx++;
        }
    }

    return VV_OK;
}
