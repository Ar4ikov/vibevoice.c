/** @file vv_queue.h
 * @brief Bounded FIFO admission for synchronous transcription requests.
 */
#ifndef VV_QUEUE_H
#define VV_QUEUE_H
#include "vv_thread.h"
#include <stdint.h>

typedef struct {
    vv_mutex_t lock;
    vv_cond_t changed;
    int active, waiting, slots, capacity;
    uint64_t next, serving, rejected;
} vv_queue_t;

static inline void vv_queue_init(vv_queue_t* q, int slots, int capacity) {
    q->active = q->waiting = 0;
    q->next = q->serving = q->rejected = 0;
    q->slots = slots; q->capacity = capacity;
    vv_mutex_init(&q->lock); vv_cond_init(&q->changed);
}
static inline bool vv_queue_enter(vv_queue_t* q) {
    vv_mutex_lock(&q->lock);
    if (q->active + q->waiting >= q->slots + q->capacity) {
        q->rejected++;
        vv_mutex_unlock(&q->lock); return false;
    }
    const uint64_t ticket = q->next++;
    q->waiting++;
    while (ticket != q->serving || q->active >= q->slots)
        vv_cond_wait(&q->changed, &q->lock);
    q->waiting--; q->active++; q->serving++;
    vv_cond_broadcast(&q->changed);
    vv_mutex_unlock(&q->lock); return true;
}
static inline void vv_queue_leave(vv_queue_t* q) {
    vv_mutex_lock(&q->lock); q->active--;
    vv_cond_broadcast(&q->changed); vv_mutex_unlock(&q->lock);
}
static inline void vv_queue_destroy(vv_queue_t* q) {
    vv_cond_destroy(&q->changed); vv_mutex_destroy(&q->lock);
}
#endif
