/**
 * @file vv_thread.h
 * @brief The threading primitives the runtime needs, on both platforms.
 *
 * Internal header: threads, a mutex, a condition variable and a one-time
 * initialiser, mapped onto pthreads or the Win32 equivalents. Not part of the
 * public API.
 *
 * `vv_once` exists for the lazily built lookup tables — the GPT-2 byte
 * alphabet, the NF4 nibble pairs. They are pure functions of constants, so
 * racing threads would write the same bytes, but a reader can still see a
 * half-built table and several inference contexts do start decoding at once.
 * One uncontended lock on a path that runs once per token is not worth
 * arguing about.
 */
#ifndef VV_THREAD_H
#define VV_THREAD_H

/*
 * Thread-local storage. C11 spells it _Thread_local, but MSVC only learned
 * that keyword in 2019 16.8 and the compiler extensions predate it
 * everywhere, so use those.
 */
#if defined(_MSC_VER)
#define VV_TLS __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define VV_TLS __thread
#else
#define VV_TLS _Thread_local
#endif

#include <stdbool.h>

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef CRITICAL_SECTION   vv_mutex_t;
typedef CONDITION_VARIABLE vv_cond_t;
typedef HANDLE             vv_thread_t;

static inline void vv_mutex_init(vv_mutex_t* m)    { InitializeCriticalSection(m); }
static inline void vv_mutex_destroy(vv_mutex_t* m) { DeleteCriticalSection(m); }
static inline void vv_mutex_lock(vv_mutex_t* m)    { EnterCriticalSection(m); }
static inline void vv_mutex_unlock(vv_mutex_t* m)  { LeaveCriticalSection(m); }

static inline void vv_cond_init(vv_cond_t* c)      { InitializeConditionVariable(c); }
static inline void vv_cond_destroy(vv_cond_t* c)   { (void)c; }
static inline void vv_cond_signal(vv_cond_t* c)    { WakeConditionVariable(c); }
static inline void vv_cond_broadcast(vv_cond_t* c) { WakeAllConditionVariable(c); }
static inline void vv_cond_wait(vv_cond_t* c, vv_mutex_t* m) {
    SleepConditionVariableCS(c, m, INFINITE);
}
/** Wait at most `ms`; false when the time ran out (spurious wakes return
 *  true, so callers loop on their condition either way). */
static inline bool vv_cond_timedwait(vv_cond_t* c, vv_mutex_t* m, int ms) {
    return SleepConditionVariableCS(c, m, (DWORD)(ms > 0 ? ms : 0)) != 0;
}

/* A flag one thread sets and another polls (a cancel request). */
typedef volatile LONG vv_atomic_int_t;
static inline void vv_atomic_store(vv_atomic_int_t* a, int v) {
    InterlockedExchange(a, (LONG)v);
}
static inline int vv_atomic_load(vv_atomic_int_t* a) {
    return (int)InterlockedCompareExchange(a, 0, 0);
}

typedef unsigned (__stdcall *vv_thread_fn)(void*);

static inline bool vv_thread_start(vv_thread_t* t, vv_thread_fn fn, void* arg) {
    *t = (HANDLE)CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)fn, arg, 0, NULL);
    return *t != NULL;
}
static inline void vv_thread_join(vv_thread_t t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}
static inline void vv_sleep_ms(int ms) { Sleep((DWORD)ms); }

#define VV_THREAD_RET      unsigned __stdcall
#define VV_THREAD_RETURN   return 0

typedef INIT_ONCE vv_once_t;
#define VV_ONCE_INIT INIT_ONCE_STATIC_INIT

static inline BOOL CALLBACK vv_once_thunk(PINIT_ONCE once, PVOID param,
                                          PVOID* ctx) {
    (void)once; (void)ctx;
    (*(void (**)(void))param)();
    return TRUE;
}
static inline void vv_once(vv_once_t* o, void (*fn)(void)) {
    InitOnceExecuteOnce(o, vv_once_thunk, &fn, NULL);
}

#else /* POSIX */

#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <unistd.h>

typedef pthread_mutex_t vv_mutex_t;
typedef pthread_cond_t  vv_cond_t;
typedef pthread_t       vv_thread_t;

static inline void vv_mutex_init(vv_mutex_t* m)    { pthread_mutex_init(m, NULL); }
static inline void vv_mutex_destroy(vv_mutex_t* m) { pthread_mutex_destroy(m); }
static inline void vv_mutex_lock(vv_mutex_t* m)    { pthread_mutex_lock(m); }
static inline void vv_mutex_unlock(vv_mutex_t* m)  { pthread_mutex_unlock(m); }

static inline void vv_cond_init(vv_cond_t* c)      { pthread_cond_init(c, NULL); }
static inline void vv_cond_destroy(vv_cond_t* c)   { pthread_cond_destroy(c); }
static inline void vv_cond_signal(vv_cond_t* c)    { pthread_cond_signal(c); }
static inline void vv_cond_broadcast(vv_cond_t* c) { pthread_cond_broadcast(c); }
static inline void vv_cond_wait(vv_cond_t* c, vv_mutex_t* m) {
    pthread_cond_wait(c, m);
}
static inline bool vv_cond_timedwait(vv_cond_t* c, vv_mutex_t* m, int ms) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    if (ms < 0) ms = 0;
    ts.tv_sec += ms / 1000;
    ts.tv_nsec += (long)(ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    return pthread_cond_timedwait(c, m, &ts) == 0;
}

typedef atomic_int vv_atomic_int_t;
static inline void vv_atomic_store(vv_atomic_int_t* a, int v) {
    atomic_store(a, v);
}
static inline int vv_atomic_load(vv_atomic_int_t* a) {
    return atomic_load(a);
}

typedef void* (*vv_thread_fn)(void*);

static inline bool vv_thread_start(vv_thread_t* t, vv_thread_fn fn, void* arg) {
    return pthread_create(t, NULL, fn, arg) == 0;
}
static inline void vv_thread_join(vv_thread_t t) { pthread_join(t, NULL); }
static inline void vv_sleep_ms(int ms) { usleep((useconds_t)ms * 1000); }

#define VV_THREAD_RET      void*
#define VV_THREAD_RETURN   return NULL

typedef pthread_once_t vv_once_t;
#define VV_ONCE_INIT PTHREAD_ONCE_INIT

static inline void vv_once(vv_once_t* o, void (*fn)(void)) {
    pthread_once(o, fn);
}

#endif

#endif /* VV_THREAD_H */
