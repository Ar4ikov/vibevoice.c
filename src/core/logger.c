/**
 * @file logger.c
 * @brief Simple logging facility for vibevoice.c
 */

#include "vibevoice/vibevoice.h"
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

static _Thread_local vv_log_level_t s_log_level = VV_LOG_INFO;

void vv_log_set_level(vv_log_level_t level) {
    s_log_level = level;
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
