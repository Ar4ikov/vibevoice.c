/**
 * @file decode.c
 * @brief Accept audio that is not a WAV file.
 *
 * The runtime's own parser handles WAV, which covers the CLI and the
 * microphone. An OpenAI-compatible endpoint has to take whatever a client
 * uploads, so anything that is not RIFF/WAVE is piped through ffmpeg when it
 * is on PATH. That keeps the dependency optional: no ffmpeg means no mp3, not
 * a broken build.
 */

#include "vibevoice/audio.h"
#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#define popen  _popen
#define pclose _pclose
#else
#include <unistd.h>
#endif

/** @brief True if the first bytes look like a RIFF/WAVE header. */
static bool is_wav(const void* data, size_t size) {
    const unsigned char* p = (const unsigned char*)data;
    return size >= 12 && memcmp(p, "RIFF", 4) == 0 && memcmp(p + 8, "WAVE", 4) == 0;
}

/**
 * @brief Decode `path` with ffmpeg into mono float32 at 24 kHz.
 *
 * ffmpeg writes raw f32le on stdout, so there is no container to parse and
 * no temporary file on the output side.
 */
static vv_status_t decode_with_ffmpeg(const char* path, float** samples,
                                      int* num_samples, int* sample_rate) {
    char cmd[1200];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -v error -nostdin -i \"%s\" -f f32le -ac 1 -ar 24000 - "
#ifndef _WIN32
             "2>/dev/null"
#endif
             , path);

    /* glibc's popen only accepts "r"/"w"; "rb" fails with EINVAL. */
#ifdef _WIN32
    FILE* pipe = popen(cmd, "rb");
#else
    FILE* pipe = popen(cmd, "r");
#endif
    if (!pipe) {
        VV_LOG_E("audio: cannot run ffmpeg (needed for non-WAV input)");
        return VV_ERR_AUDIO_FORMAT;
    }

    size_t cap = 1 << 20, len = 0;
    float* buf = (float*)vv_alloc(cap * sizeof(float));
    if (!buf) { pclose(pipe); return VV_ERR_OUT_OF_MEMORY; }

    for (;;) {
        if (len == cap) {
            size_t ncap = cap * 2;
            float* nb = (float*)vv_alloc(ncap * sizeof(float));
            if (!nb) { vv_free(buf); pclose(pipe); return VV_ERR_OUT_OF_MEMORY; }
            memcpy(nb, buf, len * sizeof(float));
            vv_free(buf);
            buf = nb;
            cap = ncap;
        }
        size_t got = fread(buf + len, sizeof(float), cap - len, pipe);
        if (got == 0) break;
        len += got;
    }
    const int rc = pclose(pipe);

    if (len == 0) {
        vv_free(buf);
        VV_LOG_E("audio: ffmpeg produced no samples (exit %d)", rc);
        return VV_ERR_AUDIO_FORMAT;
    }

    *samples = buf;
    *num_samples = (int)len;
    *sample_rate = 24000;
    VV_LOG_I("audio: decoded %s via ffmpeg -> %d samples @ 24 kHz",
             path, (int)len);
    return VV_OK;
}

vv_status_t vv_audio_load_any(const char* path, float** samples,
                              int* num_samples, int* sample_rate) {
    if (!path || !samples || !num_samples || !sample_rate)
        return VV_ERR_NULL_PTR;

    unsigned char head[12];
    FILE* f = fopen(path, "rb");
    if (!f) return VV_ERR_IO;
    const size_t got = fread(head, 1, sizeof(head), f);
    fclose(f);

    if (is_wav(head, got))
        return vv_audio_load_wav(path, samples, num_samples, sample_rate);
    return decode_with_ffmpeg(path, samples, num_samples, sample_rate);
}

vv_status_t vv_audio_load_memory(const void* data, size_t size,
                                 float** samples, int* num_samples,
                                 int* sample_rate) {
    if (!data || !samples || !num_samples || !sample_rate)
        return VV_ERR_NULL_PTR;
    if (size < 16) return VV_ERR_AUDIO_FORMAT;

    /* Both paths need a real file: the WAV parser mmaps, ffmpeg opens. */
    char path[512];
    const char* tmpdir = getenv("TMPDIR");
#ifdef _WIN32
    if (!tmpdir) tmpdir = getenv("TEMP");
    if (!tmpdir) tmpdir = ".";
#else
    if (!tmpdir) tmpdir = "/tmp";
#endif
    static unsigned counter = 0;
    snprintf(path, sizeof(path), "%s/vv_upload_%u_%u.bin", tmpdir,
             (unsigned)
#ifdef _WIN32
             GetCurrentProcessId(),
#else
             getpid(),
#endif
             counter++);

    FILE* f = fopen(path, "wb");
    if (!f) return VV_ERR_IO;
    const size_t wrote = fwrite(data, 1, size, f);
    fclose(f);
    if (wrote != size) { remove(path); return VV_ERR_IO; }

    vv_status_t s = vv_audio_load_any(path, samples, num_samples, sample_rate);
    remove(path);
    return s;
}
