/**
 * @file clock.c
 * @brief Monotonic time and sleep, on both platforms.
 */

#include "vibevoice/vibevoice.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

double vv_time_ms(void) {
    static LARGE_INTEGER freq;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
}

void vv_msleep(int ms) { Sleep((DWORD)(ms > 0 ? ms : 0)); }

#else
#include <time.h>
#include <unistd.h>

double vv_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

void vv_msleep(int ms) { if (ms > 0) usleep((useconds_t)ms * 1000); }

#endif
