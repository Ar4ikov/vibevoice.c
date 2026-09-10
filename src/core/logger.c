/**
 * @file logger.c
 * @brief Simple logging facility for vibevoice.c
 */

#include "vibevoice/vibevoice.h"
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local vv_log_level_t s_log_level = VV_LOG_INFO;

void vv_log_set_level(vv_log_level_t level) {
    s_log_level = level;
}

vv_log_level_t vv_log_get_level(void) {
    return s_log_level;
}

void vv_log(vv_log_level_t level, const char* fmt, ...) {
    if (level > s_log_level) return;

    static const char* level_names[] = {
        "ERROR", "WARN", "INFO", "DEBUG"
    };

    FILE* out = (level <= VV_LOG_WARN) ? stderr : stdout;

    /* Timestamp */
    time_t now = time(NULL);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif

    fprintf(out, "[%02d:%02d:%02d][vv][%s] ",
            tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
            level_names[level]);

    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);

    fputc('\n', out);
    fflush(out);
}


/* ─── Debug tensor dumping ──────────────────────────────────────────────── */

const char* vv_debug_dump_dir(void) {
    static const char* d = NULL;
    static int probed = 0;
    if (!probed) { d = getenv("VV_DUMP_DIR"); probed = 1; }
    return (d && d[0]) ? d : NULL;
}

void vv_debug_dump(const char* name, const void* data, size_t bytes) {
    const char* d = vv_debug_dump_dir();
    if (!d || !data || bytes == 0) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", d, name);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(data, 1, bytes, f);
    fclose(f);
}
