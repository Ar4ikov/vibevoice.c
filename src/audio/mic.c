/**
 * @file mic.c
 * @brief Microphone capture. See capture.h for the backend order.
 *
 * A capture thread fills a ring buffer; the caller drains it whenever it
 * likes. Dropping the oldest samples on overrun is the right failure mode for
 * live transcription — falling behind should cost a moment of audio, not
 * unbounded memory — and the drop count is reported so a caller can say so.
 *
 * ALSA is loaded with dlopen rather than linked. The binary has to start on a
 * machine with no libasound, and a headless server never records anything;
 * when the library is missing, an external recorder on a pipe takes over.
 *
 * Windows and macOS go through that same pipe (ffmpeg's dshow or avfoundation
 * input). A native WASAPI path would drop the ffmpeg dependency there, but it
 * is a few hundred lines of COM that cannot be tested from here, and an
 * untested capture path is worse than an honest external one.
 */

#include "vibevoice/capture.h"
#include "vibevoice/audio.h"
#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vv_thread.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

typedef enum {
    SRC_NONE = 0,
    SRC_ALSA,
    SRC_PIPE,
    SRC_FILE,
} src_kind_t;

struct vv_mic {
    src_kind_t kind;
    int        sample_rate;
    const char* backend_name;
    char       device_label[256];

    /* ring buffer */
    float*     ring;
    int        cap;
    volatile int head;      /* written by the capture thread */
    volatile int tail;      /* read by the consumer          */
    uint64_t    overruns;
    uint64_t    captured;

    vv_thread_t thread;
    volatile int running;
    volatile int eof;

    /* source-specific */
    void*      alsa_handle;   /* snd_pcm_t*            */
    void*      alsa_lib;
    FILE*      pipe;
    float*     file_pcm;      /* whole file, for SRC_FILE */
    int        file_len;
    int        file_pos;
    bool       file_realtime;
};

/* ─── Ring buffer ────────────────────────────────────────────────────────── */

static void ring_write(vv_mic_t* m, const float* src, int n) {
    m->captured += (uint64_t)n;
    for (int i = 0; i < n; i++) {
        const int nh = (m->head + 1) % m->cap;
        if (nh == m->tail) {          /* full: drop the oldest sample */
            m->tail = (m->tail + 1) % m->cap;
            m->overruns++;
        }
        m->ring[m->head] = src[i];
        m->head = nh;
    }
}

int vv_mic_read(vv_mic_t* m, float* dst, int max) {
    if (!m || !dst || max <= 0) return 0;
    int n = 0;
    while (n < max && m->tail != m->head) {
        dst[n++] = m->ring[m->tail];
        m->tail = (m->tail + 1) % m->cap;
    }
    return n;
}

bool vv_mic_eof(const vv_mic_t* m) {
    return m && m->eof && m->tail == m->head;
}

uint64_t vv_mic_overruns(const vv_mic_t* m) { return m ? m->overruns : 0; }

const char* vv_mic_backend(const vv_mic_t* m) {
    return m && m->backend_name ? m->backend_name : "none";
}

const char* vv_mic_device_label(const vv_mic_t* m) {
    return m ? m->device_label : "";
}

uint64_t vv_mic_captured(const vv_mic_t* m) { return m ? m->captured : 0; }

/* ─── ALSA, loaded at runtime ────────────────────────────────────────────── */

#ifndef _WIN32
typedef int (*fn_open)(void**, const char*, int, int);
typedef int (*fn_set_params)(void*, int, int, unsigned, unsigned, int, unsigned);
typedef long (*fn_readi)(void*, void*, unsigned long);
typedef int (*fn_close)(void*);
typedef int (*fn_recover)(void*, int, int);

static struct {
    void* lib;
    fn_open       open;
    fn_set_params set_params;
    fn_readi      readi;
    fn_close      close;
    fn_recover    recover;
} g_alsa;

/* From alsa/pcm.h; stable ABI values. */
#define SND_PCM_STREAM_CAPTURE   1
#define SND_PCM_FORMAT_FLOAT_LE  14
#define SND_PCM_ACCESS_RW_INTERLEAVED 3

static bool alsa_load(void) {
    if (g_alsa.lib) return true;
    void* lib = dlopen("libasound.so.2", RTLD_LAZY);
    if (!lib) lib = dlopen("libasound.so", RTLD_LAZY);
    if (!lib) return false;

    g_alsa.open       = (fn_open)dlsym(lib, "snd_pcm_open");
    g_alsa.set_params = (fn_set_params)dlsym(lib, "snd_pcm_set_params");
    g_alsa.readi      = (fn_readi)dlsym(lib, "snd_pcm_readi");
    g_alsa.close      = (fn_close)dlsym(lib, "snd_pcm_close");
    g_alsa.recover    = (fn_recover)dlsym(lib, "snd_pcm_recover");
    if (!g_alsa.open || !g_alsa.set_params || !g_alsa.readi || !g_alsa.close) {
        dlclose(lib);
        memset(&g_alsa, 0, sizeof(g_alsa));
        return false;
    }
    g_alsa.lib = lib;
    return true;
}

static VV_THREAD_RET alsa_thread(void* arg) {
    vv_mic_t* m = (vv_mic_t*)arg;
    const int chunk = m->sample_rate / 50;          /* 20 ms */
    float* buf = (float*)vv_alloc((size_t)chunk * sizeof(float));
    if (!buf) { m->eof = 1; VV_THREAD_RETURN; }

    while (m->running) {
        long got = g_alsa.readi(m->alsa_handle, buf, (unsigned long)chunk);
        if (got < 0) {
            if (g_alsa.recover && g_alsa.recover(m->alsa_handle, (int)got, 1) == 0)
                continue;
            break;
        }
        if (got > 0) ring_write(m, buf, (int)got);
    }
    vv_free(buf);
    m->eof = 1;
    VV_THREAD_RETURN;
}
#endif /* !_WIN32 */

/* ─── External recorder on a pipe ────────────────────────────────────────── */

static VV_THREAD_RET pipe_thread(void* arg) {
    vv_mic_t* m = (vv_mic_t*)arg;
    const int chunk = m->sample_rate / 50;
    float* buf = (float*)vv_alloc((size_t)chunk * sizeof(float));
    if (!buf) { m->eof = 1; VV_THREAD_RETURN; }

    while (m->running) {
        const size_t got = fread(buf, sizeof(float), (size_t)chunk, m->pipe);
        if (got == 0) break;
        ring_write(m, buf, (int)got);
    }
    vv_free(buf);
    m->eof = 1;
    VV_THREAD_RETURN;
}

/* ─── File playback at wall-clock speed ──────────────────────────────────── */

static VV_THREAD_RET file_thread(void* arg) {
    vv_mic_t* m = (vv_mic_t*)arg;
    const int chunk = m->sample_rate / 50;          /* 20 ms */

    while (m->running && m->file_pos < m->file_len) {
        const int n = (m->file_len - m->file_pos < chunk)
                      ? (m->file_len - m->file_pos) : chunk;
        ring_write(m, m->file_pcm + m->file_pos, n);
        m->file_pos += n;
        if (m->file_realtime) vv_sleep_ms(1000 * n / m->sample_rate);
    }
    m->eof = 1;
    VV_THREAD_RETURN;
}

/* ─── Open / close ───────────────────────────────────────────────────────── */

static vv_status_t mic_alloc(int sample_rate, vv_mic_t** out) {
    vv_mic_t* m = (vv_mic_t*)vv_alloc(sizeof(vv_mic_t));
    if (!m) return VV_ERR_OUT_OF_MEMORY;
    memset(m, 0, sizeof(*m));
    m->sample_rate = sample_rate;
    m->cap = sample_rate * 30;                  /* 30 s of slack */
    m->ring = (float*)vv_alloc((size_t)m->cap * sizeof(float));
    if (!m->ring) { vv_free(m); return VV_ERR_OUT_OF_MEMORY; }
    m->running = 1;
    *out = m;
    return VV_OK;
}

/** @brief Command line for the fallback recorder, or NULL if none is usable. */
static const char* recorder_cmd(int rate, const char* device, char* buf,
                                size_t n) {
#ifdef __APPLE__
    snprintf(buf, n, "ffmpeg -v error -f avfoundation -i \":%s\" "
                     "-f f32le -ac 1 -ar %d - 2>/dev/null",
             device && device[0] ? device : "0", rate);
#elif defined(_WIN32)
    /*
     * dshow has no device called "default", so an unresolved name is a hard
     * error rather than a fallback -- vv_mic_open resolves it first.
     */
    snprintf(buf, n, "ffmpeg -v error -f dshow -i audio=\"%s\" "
                     "-f f32le -ac 1 -ar %d -",
             device && device[0] ? device : "default", rate);
#else
    snprintf(buf, n, "arecord -q -f FLOAT_LE -c 1 -r %d -t raw -D %s 2>/dev/null",
             rate, device && device[0] ? device : "default");
#endif
    return buf;
}

vv_status_t vv_mic_open(int sample_rate, const char* device, vv_mic_t** out) {
    if (!out) return VV_ERR_NULL_PTR;
    if (sample_rate <= 0) sample_rate = 24000;

    vv_mic_t* m = NULL;
    vv_status_t s = mic_alloc(sample_rate, &m);
    if (s != VV_OK) return s;

#ifndef _WIN32
    if (alsa_load()) {
        void* h = NULL;
        const char* dev = (device && device[0]) ? device : "default";
        if (g_alsa.open(&h, dev, SND_PCM_STREAM_CAPTURE, 0) == 0) {
            const int rc = g_alsa.set_params(
                h, SND_PCM_FORMAT_FLOAT_LE, SND_PCM_ACCESS_RW_INTERLEAVED,
                1, (unsigned)sample_rate, 1, 100000 /* 100 ms latency */);
            if (rc == 0) {
                m->alsa_handle = h;
                m->kind = SRC_ALSA;
                m->backend_name = "ALSA";
                snprintf(m->device_label, sizeof(m->device_label), "%s", dev);
                if (vv_thread_start(&m->thread, (vv_thread_fn)alsa_thread, m)) {
                    *out = m;
                    return VV_OK;
                }
            }
            g_alsa.close(h);
        }
    }
#endif

    /* Fall back to an external recorder. */
    {
        char cmd[1024];
        char resolved[384];
        if (!vv_mic_resolve_device(device, resolved, sizeof(resolved),
                                   m->device_label,
                                   sizeof(m->device_label))) {
            VV_LOG_E("mic: no capture device found "
                     "(run `vv_cli devices`; on Windows this needs ffmpeg "
                     "on PATH)");
            vv_free(m->ring);
            vv_free(m);
            return VV_ERR_NOT_FOUND;
        }
        recorder_cmd(sample_rate, resolved, cmd, sizeof(cmd));
#ifdef _WIN32
        m->pipe = _popen(cmd, "rb");
#else
        m->pipe = popen(cmd, "r");
#endif
        if (m->pipe) {
            m->kind = SRC_PIPE;
            m->backend_name =
#ifdef __linux__
                "arecord";
#else
                "ffmpeg";
#endif
            if (vv_thread_start(&m->thread, (vv_thread_fn)pipe_thread, m)) {
                *out = m;
                return VV_OK;
            }
        }
    }

    VV_LOG_E("mic: no capture backend available "
             "(tried the system audio API, then an external recorder)");
    vv_free(m->ring);
    vv_free(m);
    return VV_ERR_NOT_FOUND;
}

vv_status_t vv_mic_open_file(const char* wav_path, int sample_rate,
                             bool realtime, vv_mic_t** out) {
    if (!wav_path || !out) return VV_ERR_NULL_PTR;
    if (sample_rate <= 0) sample_rate = 24000;

    float* raw = NULL;
    int n = 0, sr = 0;
    vv_status_t s = vv_audio_load_any(wav_path, &raw, &n, &sr);
    if (s != VV_OK) return s;

    float* pcm = raw;
    int len = n;
    if (sr != sample_rate) {
        float* rs = NULL;
        int rn = 0;
        s = vv_audio_resample(raw, sr, n, &rs, sample_rate, &rn);
        vv_free(raw);
        if (s != VV_OK) return s;
        pcm = rs;
        len = rn;
    }

    vv_mic_t* m = NULL;
    s = mic_alloc(sample_rate, &m);
    if (s != VV_OK) { vv_free(pcm); return s; }

    m->kind = SRC_FILE;
    m->backend_name = realtime ? "file (real time)" : "file";
    m->file_pcm = pcm;
    m->file_len = len;
    m->file_realtime = realtime;

    if (!vv_thread_start(&m->thread, (vv_thread_fn)file_thread, m)) {
        vv_free(pcm);
        vv_free(m->ring);
        vv_free(m);
        return VV_ERR_IO;
    }
    *out = m;
    return VV_OK;
}

void vv_mic_close(vv_mic_t* m) {
    if (!m) return;
    m->running = 0;

    if (m->pipe) {
#ifdef _WIN32
        _pclose(m->pipe);
#else
        pclose(m->pipe);
#endif
        m->pipe = NULL;
    }
    vv_thread_join(m->thread);

#ifndef _WIN32
    if (m->alsa_handle && g_alsa.close) g_alsa.close(m->alsa_handle);
#endif
    vv_free(m->file_pcm);
    vv_free(m->ring);
    vv_free(m);
}
