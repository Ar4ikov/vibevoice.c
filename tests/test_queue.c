#include "vv_queue.h"
#include <stdio.h>

typedef struct { vv_queue_t* q; int id; int* order; int* count; } task_t;
static VV_THREAD_RET run(void* p) {
    task_t* t = p;
    if (!vv_queue_enter(t->q)) VV_THREAD_RETURN;
    t->order[(*t->count)++] = t->id;
    vv_queue_leave(t->q);
    VV_THREAD_RETURN;
}
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "failed: %s\n", #x); return 1; } } while (0)
int main(void) {
    vv_queue_t q;
    vv_queue_init(&q, 1, 2);
    CHECK(vv_queue_enter(&q));
    int order[2] = {-1, -1}, count = 0;
    task_t tasks[2] = {{&q, 0, order, &count}, {&q, 1, order, &count}};
    vv_thread_t threads[2];
    for (int i = 0; i < 2; i++) {
        CHECK(vv_thread_start(&threads[i], run, &tasks[i]));
        int waiting = 0;
        for (int tries = 0; tries < 5000; tries++) {
            vv_mutex_lock(&q.lock); waiting = q.waiting; vv_mutex_unlock(&q.lock);
            if (waiting == i + 1) break;
            vv_sleep_ms(1);
        }
        CHECK(waiting == i + 1);
    }
    CHECK(!vv_queue_enter(&q));
    vv_queue_leave(&q);
    for (int i = 0; i < 2; i++) vv_thread_join(threads[i]);
    CHECK(count == 2 && order[0] == 0 && order[1] == 1);
    CHECK(q.active == 0 && q.waiting == 0 && q.rejected == 1);
    CHECK(vv_queue_enter(&q)); vv_queue_leave(&q);
    vv_queue_destroy(&q);
    vv_queue_init(&q, 2, 0);
    CHECK(vv_queue_enter(&q)); CHECK(vv_queue_enter(&q));
    CHECK(!vv_queue_enter(&q));
    vv_queue_leave(&q); CHECK(vv_queue_enter(&q));
    vv_queue_leave(&q); vv_queue_leave(&q); vv_queue_destroy(&q);

    /* try_enter: in when a slot is free, out after the wait when not. */
    vv_queue_init(&q, 1, 4);
    CHECK(vv_queue_try_enter(&q, 0));
    const double t0 = vv_time_ms();
    CHECK(!vv_queue_try_enter(&q, 50));
    const double waited = vv_time_ms() - t0;
    CHECK(waited >= 40 && waited < 2000);
    CHECK(q.rejected == 1);
    vv_queue_leave(&q);
    CHECK(vv_queue_try_enter(&q, 0));
    /* ... and never ahead of a request already queued for the slot. */
    count = 0;
    order[0] = order[1] = -1;
    vv_thread_t th;
    CHECK(vv_thread_start(&th, run, &tasks[0]));
    for (int tries = 0; tries < 5000; tries++) {
        int waiting = 0;
        vv_mutex_lock(&q.lock); waiting = q.waiting; vv_mutex_unlock(&q.lock);
        if (waiting == 1) break;
        vv_sleep_ms(1);
    }
    vv_queue_leave(&q);
    vv_thread_join(th);
    CHECK(count == 1);
    CHECK(q.active == 0);
    CHECK(vv_queue_try_enter(&q, 0));
    vv_queue_leave(&q);
    vv_queue_destroy(&q);
    puts("queue: FIFO, overflow, reuse, zero-capacity and try_enter passed");
    return 0;
}
