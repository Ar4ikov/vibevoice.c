/**
 * @file vv_thread.h
 * @brief The three threading primitives the runtime needs, on both platforms.
 *
 * Internal header: threads, a mutex and a condition variable, mapped onto
 * pthreads or the Win32 equivalents. Not part of the public API.
 */
#ifndef VV_THREAD_H
#define VV_THREAD_H

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

#else /* POSIX */

#include <pthread.h>
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

typedef void* (*vv_thread_fn)(void*);

static inline bool vv_thread_start(vv_thread_t* t, vv_thread_fn fn, void* arg) {
    return pthread_create(t, NULL, fn, arg) == 0;
}
static inline void vv_thread_join(vv_thread_t t) { pthread_join(t, NULL); }
static inline void vv_sleep_ms(int ms) { usleep((useconds_t)ms * 1000); }

#define VV_THREAD_RET      void*
#define VV_THREAD_RETURN   return NULL

#endif

#endif /* VV_THREAD_H */
