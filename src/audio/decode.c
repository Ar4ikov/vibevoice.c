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
#include "vv_spawn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
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
 * no temporary file on the output side. It is started from an argument
 * vector, never through a shell, and the input is given as `file:<path>`:
 * whatever the path contains, it is a local file name, not a shell word, an
 * option, or a protocol such as `concat:` or `http:`.
 */
static vv_status_t decode_with_ffmpeg(const char* path, float** samples,
                                      int* num_samples, int* sample_rate) {
    const size_t plen = strlen(path);
    char* input = (char*)vv_alloc(plen + 6);
    if (!input) return VV_ERR_OUT_OF_MEMORY;
    memcpy(input, "file:", 5);
    memcpy(input + 5, path, plen + 1);

    const char* const argv[] = {
        "ffmpeg", "-v", "error", "-nostdin", "-i", input,
        "-f", "f32le", "-ac", "1", "-ar", "24000", "pipe:1", NULL
    };
    FILE* pipe = NULL;
    vv_child_t* child = vv_spawn_read(argv, VV_SPAWN_STDERR_NULL, &pipe);
    vv_free(input);
    if (!child) {
        VV_LOG_E("audio: cannot run ffmpeg (needed for non-WAV input)");
        return VV_ERR_AUDIO_FORMAT;
    }

    size_t cap = 1 << 20, len = 0;
    float* buf = (float*)vv_alloc(cap * sizeof(float));
    if (!buf) { vv_spawn_wait(child, true); return VV_ERR_OUT_OF_MEMORY; }

    for (;;) {
        if (len == cap) {
            size_t ncap = cap * 2;
            float* nb = (float*)vv_alloc(ncap * sizeof(float));
            if (!nb) { vv_free(buf); vv_spawn_wait(child, true); return VV_ERR_OUT_OF_MEMORY; }
            memcpy(nb, buf, len * sizeof(float));
            vv_free(buf);
            buf = nb;
            cap = ncap;
        }
        size_t got = fread(buf + len, sizeof(float), cap - len, pipe);
        if (got == 0) break;
        len += got;
    }
    const int rc = vv_spawn_wait(child, false);

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

/**
 * @brief Create a new, empty, private file for an upload and open it.
 *
 * The name is chosen by the OS and the file is created exclusively, so no
 * two requests can share one and nothing placed in the temp directory in
 * advance -- a symlink, a file owned by someone else -- is ever opened in its
 * place. POSIX: mkstemp() in $TMPDIR (when absolute) or /tmp, mode 0600.
 * Windows: GetTempFileName() in the user's temp directory.
 */
static FILE* create_upload_file(char* path, size_t path_size) {
#ifdef _WIN32
    char dir[MAX_PATH + 1];
    const DWORD n = GetTempPathA((DWORD)sizeof(dir), dir);
    if (n == 0 || n > sizeof(dir)) return NULL;
    char name[MAX_PATH + 1];
    if (!GetTempFileNameA(dir, "vvu", 0, name)) return NULL;
    if (strlen(name) + 1 > path_size) { DeleteFileA(name); return NULL; }
    strcpy(path, name);
    FILE* f = fopen(path, "wb");
    if (!f) DeleteFileA(path);
    return f;
#else
    const char* dir = getenv("TMPDIR");
    if (!dir || dir[0] != '/') dir = "/tmp";
    const int n = snprintf(path, path_size, "%s/vv_upload_XXXXXX", dir);
    if (n <= 0 || (size_t)n >= path_size) return NULL;
    const int fd = mkstemp(path);
    if (fd < 0) return NULL;
    FILE* f = fdopen(fd, "wb");
    if (!f) { close(fd); unlink(path); }
    return f;
#endif
}

vv_status_t vv_audio_load_memory(const void* data, size_t size,
                                 float** samples, int* num_samples,
                                 int* sample_rate) {
    if (!data || !samples || !num_samples || !sample_rate)
        return VV_ERR_NULL_PTR;
    if (size < 16) return VV_ERR_AUDIO_FORMAT;

    /* Both paths need a real file: the WAV parser mmaps, ffmpeg opens. */
    char path[1024];
    FILE* f = create_upload_file(path, sizeof(path));
    if (!f) {
        VV_LOG_E("audio: cannot create a temporary file for the upload");
        return VV_ERR_IO;
    }
    const size_t wrote = fwrite(data, 1, size, f);
    const int closed = fclose(f);
    if (wrote != size || closed != 0) { remove(path); return VV_ERR_IO; }

    vv_status_t s = vv_audio_load_any(path, samples, num_samples, sample_rate);
    remove(path);
    return s;
}
