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
    puts("queue: FIFO, overflow, reuse and zero-capacity passed");
    return 0;
}
