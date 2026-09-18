/**
 * @file mic_devices.c
 * @brief Enumerating capture devices, and resolving a user-supplied name.
 *
 * There is no portable way to ask "what can record?", so each platform gets
 * the cheapest honest answer:
 *
 *   Windows   ffmpeg's dshow enumerator. Friendly names are ambiguous -- a
 *             headset shows up once per endpoint and two identical headsets
 *             share a name -- so the alternative name (`@device_cm_{...}`) is
 *             what actually gets passed back to ffmpeg.
 *   macOS     ffmpeg's avfoundation enumerator; the index is the id.
 *   Linux     ALSA device hints, which is what `arecord -L` prints.
 *
 * The same list drives `vv_cli devices` and the --device lookup, so whatever
 * the user sees listed is something they can type.
 */

#include "vibevoice/capture.h"
#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "vv_spawn.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

#define VV_MAX_DEVICES 64

static void copy_field(char* dst, size_t n, const char* src) {
    snprintf(dst, n, "%s", src ? src : "");
}

/** @brief Strip the trailing newline and any trailing blanks, in place. */
static void chomp(char* s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ' ||
                 s[n - 1] == '\t'))
        s[--n] = '\0';
}

/** @brief Copy the text between the first and last double quote of `s`. */
static bool quoted(const char* s, char* out, size_t n) {
    const char* a = strchr(s, '"');
    const char* b = a ? strrchr(s, '"') : NULL;
    if (!a || !b || b <= a + 1) return false;
    size_t len = (size_t)(b - a - 1);
    if (len >= n) len = n - 1;
    memcpy(out, a + 1, len);
    out[len] = '\0';
    return true;
}

static bool all_digits(const char* s) {
    if (!s || !*s) return false;
    for (; *s; s++) if (!isdigit((unsigned char)*s)) return false;
    return true;
}

/** @brief Case-insensitive substring search (strcasestr is not portable). */
static bool contains_ci(const char* hay, const char* needle) {
    if (!hay || !needle || !*needle) return false;
    const size_t nl = strlen(needle);
    for (; *hay; hay++) {
        size_t i = 0;
        while (i < nl && hay[i] &&
               tolower((unsigned char)hay[i]) == tolower((unsigned char)needle[i]))
            i++;
        if (i == nl) return true;
    }
    return false;
}

/* --- Windows: ffmpeg -list_devices --------------------------------------- */

#if defined(_WIN32)

static int enumerate(vv_mic_device_t* d, int max) {
    /* ffmpeg prints the list on stderr, so it is merged into the pipe. */
    static const char* const argv[] = {
        "ffmpeg", "-hide_banner", "-list_devices", "true",
        "-f", "dshow", "-i", "dummy", NULL
    };
    FILE* p = NULL;
    vv_child_t* child = vv_spawn_read(argv, VV_SPAWN_STDERR_MERGE, &p);
    if (!child) return 0;

    char line[1024];
    int n = 0;
    int section_audio = 0;     /* which header we are under */
    int last_is_audio = 0;     /* did the previous name line make the cut */

    while (fgets(line, sizeof(line), p)) {
        char* s = line;
        if (*s == '[') {                       /* drop the "[dshow @ ..] " tag */
            char* e = strstr(s, "] ");
            if (e) s = e + 2;
        }
        while (*s == ' ' || *s == '\t') s++;
        chomp(s);
        if (!*s) continue;

        if (strstr(s, "video devices")) { section_audio = 0; last_is_audio = 0; continue; }
        if (strstr(s, "audio devices")) { section_audio = 1; last_is_audio = 0; continue; }

        if (strncmp(s, "Alternative name", 16) == 0) {
            char alt[sizeof(d[0].id)];
            if (last_is_audio && n > 0 && quoted(s, alt, sizeof(alt)))
                copy_field(d[n - 1].id, sizeof(d[n - 1].id), alt);
            continue;
        }

        if (*s == '"') {
            char name[sizeof(d[0].name)];
            if (!quoted(s, name, sizeof(name))) continue;
            int is_audio = section_audio;
            if (strstr(s, "(audio)")) is_audio = 1;
            else if (strstr(s, "(video)")) is_audio = 0;
            last_is_audio = is_audio && n < max;
            if (!last_is_audio) continue;
            memset(&d[n], 0, sizeof(d[n]));
            copy_field(d[n].name, sizeof(d[n].name), name);
            copy_field(d[n].id, sizeof(d[n].id), name);   /* until the alt name */
            n++;
        }
    }
    vv_spawn_wait(child, false);
    return n;
}

/* --- macOS: ffmpeg -f avfoundation -list_devices ------------------------- */

#elif defined(__APPLE__)

static int enumerate(vv_mic_device_t* d, int max) {
    /* ffmpeg prints the list on stderr, so it is merged into the pipe. */
    static const char* const argv[] = {
        "ffmpeg", "-hide_banner", "-f", "avfoundation",
        "-list_devices", "true", "-i", "", NULL
    };
    FILE* p = NULL;
    vv_child_t* child = vv_spawn_read(argv, VV_SPAWN_STDERR_MERGE, &p);
    if (!child) return 0;

    char line[1024];
    int n = 0, section_audio = 0;
    while (fgets(line, sizeof(line), p)) {
        char* s = line;
        if (*s == '[') {
            char* e = strstr(s, "] ");
            if (e) s = e + 2;
        }
        while (*s == ' ' || *s == '\t') s++;
        chomp(s);
        if (!*s) continue;

        if (strstr(s, "video devices")) { section_audio = 0; continue; }
        if (strstr(s, "audio devices")) { section_audio = 1; continue; }
        if (!section_audio || *s != '[') continue;

        /* "[0] Built-in Microphone" */
        char* close_br = strchr(s, ']');
        if (!close_br) continue;
        *close_br = '\0';
        const char* idx = s + 1;
        const char* name = close_br + 1;
        while (*name == ' ') name++;
        if (n >= max) break;
        memset(&d[n], 0, sizeof(d[n]));
        copy_field(d[n].id, sizeof(d[n].id), idx);
        copy_field(d[n].name, sizeof(d[n].name), name);
        n++;
    }
    vv_spawn_wait(child, false);
    return n;
}

/* --- Linux: ALSA device hints -------------------------------------------- */

#else

typedef int   (*fn_hint)(int, const char*, void***);
typedef char* (*fn_get_hint)(const void*, const char*);
typedef int   (*fn_free_hint)(void**);

static int enumerate(vv_mic_device_t* d, int max) {
    void* lib = dlopen("libasound.so.2", RTLD_LAZY);
    if (!lib) lib = dlopen("libasound.so", RTLD_LAZY);
    if (!lib) return 0;

    fn_hint      hint      = (fn_hint)dlsym(lib, "snd_device_name_hint");
    fn_get_hint  get_hint  = (fn_get_hint)dlsym(lib, "snd_device_name_get_hint");
    fn_free_hint free_hint = (fn_free_hint)dlsym(lib, "snd_device_name_free_hint");
    if (!hint || !get_hint || !free_hint) { dlclose(lib); return 0; }

    void** hints = NULL;
    if (hint(-1, "pcm", &hints) != 0 || !hints) { dlclose(lib); return 0; }

    int n = 0;
    for (void** h = hints; *h && n < max; h++) {
        char* io   = get_hint(*h, "IOID");
        char* name = get_hint(*h, "NAME");
        char* desc = get_hint(*h, "DESC");
        /* IOID is NULL for duplex devices, "Input" for capture-only ones. */
        const bool capture = (!io || strcmp(io, "Input") == 0);
        if (capture && name) {
            memset(&d[n], 0, sizeof(d[n]));
            copy_field(d[n].id, sizeof(d[n].id), name);
            /* The description's first line is the human-readable label. */
            if (desc) {
                char* nl = strchr(desc, '\n');
                if (nl) *nl = '\0';
            }
            copy_field(d[n].name, sizeof(d[n].name), desc ? desc : name);
            n++;
        }
        /* These come from libasound's own allocator, so they take free(). */
        free(io); free(name); free(desc);
    }
    free_hint(hints);
    dlclose(lib);
    return n;
}

#endif

/* --- Public API ---------------------------------------------------------- */

vv_status_t vv_mic_list_devices(vv_mic_device_t** out, int* count) {
    if (!out || !count) return VV_ERR_NULL_PTR;
    *out = NULL;
    *count = 0;

    vv_mic_device_t* tmp =
        (vv_mic_device_t*)vv_alloc(VV_MAX_DEVICES * sizeof(vv_mic_device_t));
    if (!tmp) return VV_ERR_OUT_OF_MEMORY;
    memset(tmp, 0, VV_MAX_DEVICES * sizeof(vv_mic_device_t));

    const int n = enumerate(tmp, VV_MAX_DEVICES);
    if (n <= 0) {
        vv_free(tmp);
        return VV_ERR_NOT_FOUND;
    }
    tmp[0].is_default = true;
    *out = tmp;
    *count = n;
    return VV_OK;
}

bool vv_mic_resolve_device(const char* request, char* out, size_t n,
                           char* label, size_t label_n) {
    if (!out || n == 0) return false;
    out[0] = '\0';
    if (label && label_n) label[0] = '\0';

    const bool want_default = !request || !request[0] ||
                              strcmp(request, "default") == 0;

#ifndef _WIN32
    /* ALSA and arecord both understand "default"; no lookup needed. */
    if (want_default) {
        copy_field(out, n, "default");
        if (label) copy_field(label, label_n, "default");
        return true;
    }
#endif

    vv_mic_device_t* devs = NULL;
    int count = 0;
    if (vv_mic_list_devices(&devs, &count) != VV_OK) {
        if (want_default) return false;       /* nothing to guess from */
        copy_field(out, n, request);          /* trust the user's spelling */
        if (label) copy_field(label, label_n, request);
        return true;
    }

    int pick = -1;
    if (want_default) {
        pick = 0;
    } else if (all_digits(request)) {
        const int idx = atoi(request);
        if (idx >= 0 && idx < count) pick = idx;
    }
    if (pick < 0) {
        for (int i = 0; i < count && pick < 0; i++)
            if (strcmp(devs[i].id, request) == 0 ||
                strcmp(devs[i].name, request) == 0) pick = i;
    }
    if (pick < 0) {
        for (int i = 0; i < count && pick < 0; i++)
            if (contains_ci(devs[i].name, request) ||
                contains_ci(devs[i].id, request)) pick = i;
    }

    if (pick < 0) {
        VV_LOG_W("mic: no capture device matches '%s'; passing it through "
                 "(run `vv_cli devices` for the list)", request);
        copy_field(out, n, request);
        if (label) copy_field(label, label_n, request);
    } else {
        copy_field(out, n, devs[pick].id);
        if (label) copy_field(label, label_n, devs[pick].name);
        VV_LOG_I("mic: using capture device '%s'", devs[pick].name);
    }
    vv_free(devs);
    return true;
}
