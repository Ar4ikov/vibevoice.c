/**
 * @file trt_utils.c
 * @brief TensorRT utility functions: error handling, logging.
 */

#include "vibevoice/trt.h"
#include "vibevoice/vibevoice.h"

#ifdef VV_HAS_TRT

/**
 * @brief Check if TRT engine files exist for both encoders.
 *
 * @param acoustic_plan Path to acoustic encoder .plan (can be NULL)
 * @param semantic_plan Path to semantic encoder .plan (can be NULL)
 * @return true if both files exist
 */
bool vv_trt_engines_available(const char* acoustic_plan,
                               const char* semantic_plan) {
    if (!acoustic_plan || !semantic_plan) return false;

    FILE* f1 = fopen(acoustic_plan, "rb");
    if (!f1) return false;
    fclose(f1);

    FILE* f2 = fopen(semantic_plan, "rb");
    if (!f2) return false;
    fclose(f2);

    return true;
}

#endif /* VV_HAS_TRT */
