/**
 * @file version.c
 * @brief What this binary says it is.
 *
 * Two things are worth asking a binary. Its version is a full SemVer 2.0
 * string: `0.2.0` for a build of the v0.2.0 tag, `0.2.1-dev.5+g1a2b3c4` for
 * one five commits past it. Its revision is the commit it was built from.
 * Both are derived by cmake/Version.cmake and stamped into a generated header
 * on every build, and the container build passes them in explicitly because
 * its context has no .git. Either way they are compiled in, so the answer is
 * the binary's own and cannot be overridden from the environment:
 *
 *     docker run --rm ghcr.io/ar4ikov/vibevoice.c:0.2.0 --version
 *     vibevoice.c 0.2.0 (1a2b3c4) [cuda openmp]
 */
#include "vibevoice/vibevoice.h"

#include "vv_build_info.h"

#ifndef VV_VERSION_FULL
#define VV_VERSION_FULL VV_VERSION_STRING
#endif
#ifndef VV_REVISION
#define VV_REVISION "unknown"
#endif

const char* vv_version(void) {
    return VV_VERSION_FULL;
}

const char* vv_build_ref(void) {
    return VV_REVISION;
}

const char* vv_build_features(void) {
    return
#if defined(VV_HAS_CUDA)
        "cuda"
#elif defined(VV_HAS_METAL)
        "metal"
#else
        "cpu-only"
#endif
#ifdef _OPENMP
        " openmp"
#endif
        ;  /* NOLINT: the parts above concatenate into one literal */
}
