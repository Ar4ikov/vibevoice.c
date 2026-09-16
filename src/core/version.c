/**
 * @file version.c
 * @brief What this binary says it is.
 *
 * Two different things are worth asking a binary. The release version is
 * compiled in and changes only when someone bumps the header. The build ref
 * is whatever the build knew about its source -- a `git describe` for a
 * working tree, the published image tag for a container -- and it is what
 * makes an image tag checkable from the inside:
 *
 *     docker run --rm ghcr.io/ar4ikov/vibevoice.c:0.1.0 --version
 *
 * The container stamps its ref through the environment rather than the
 * compile, because a build argument that reaches the `cmake` layer would
 * invalidate the build cache on every commit and rebuild all five CUDA
 * architectures for a string.
 */
#include "vibevoice/vibevoice.h"

#include <stdlib.h>

#ifndef VV_BUILD_REF
#define VV_BUILD_REF "unknown"
#endif

/** @brief True for the characters a ref is allowed to contain. */
static int ref_char_ok(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') || c == '.' || c == '-' || c == '_' ||
           c == '+';
}

const char* vv_version(void) {
    return VV_VERSION_STRING;
}

const char* vv_build_ref(void) {
    /*
     * An override has to survive being printed into JSON and into a
     * Prometheus label, so anything outside a tag's alphabet is refused
     * rather than escaped.
     */
    const char* env = getenv("VV_BUILD_REF");
    if (env && *env) {
        int n = 0;
        for (; env[n]; n++) {
            if (n >= 64 || !ref_char_ok(env[n])) return VV_BUILD_REF;
        }
        return env;
    }
    return VV_BUILD_REF;
}

const char* vv_build_features(void) {
    return
#ifdef VV_HAS_CUDA
        "cuda"
#else
        "cpu-only"
#endif
#ifdef _OPENMP
        " openmp"
#endif
        ;  /* NOLINT: the parts above concatenate into one literal */
}
