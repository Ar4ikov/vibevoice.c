/**
 * @file profiler.h
 * @brief Profiling and benchmarking utilities.
 *
 * Uses CUDA Events for GPU timing and NVTX markers for Nsight profiling.
 */
#ifndef VV_PROFILER_H
#define VV_PROFILER_H

#include "vibevoice/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Profiler context for timing measurements. */
typedef struct vv_profiler {
    void*   start_event;   /**< cudaEvent_t */
    void*   stop_event;    /**< cudaEvent_t */
    void*   stream;        /**< Associated CUDA stream */
    float   last_ms;       /**< Last measured time in ms */
    double  total_ms;      /**< Accumulated total time */
    int     n_measurements; /**< Number of measurements */
} vv_profiler_t;

/**
 * @brief Create a profiler.
 */
vv_status_t vv_profiler_create(vv_profiler_t** profiler, void* stream);

/**
 * @brief Start timing.
 */
vv_status_t vv_profiler_start(vv_profiler_t* profiler);

/**
 * @brief Stop timing and record elapsed time.
 */
vv_status_t vv_profiler_stop(vv_profiler_t* profiler);

/**
 * @brief Get last measurement in milliseconds.
 */
float vv_profiler_last_ms(const vv_profiler_t* profiler);

/**
 * @brief Get average measurement in milliseconds.
 */
float vv_profiler_avg_ms(const vv_profiler_t* profiler);

/**
 * @brief Reset accumulated statistics.
 */
vv_status_t vv_profiler_reset(vv_profiler_t* profiler);

/**
 * @brief Print profiler report to log.
 */
vv_status_t vv_profiler_report(const vv_profiler_t* profiler,
                                const char* label);

/**
 * @brief Free profiler.
 */
vv_status_t vv_profiler_free(vv_profiler_t* profiler);

/* ─── CUDA Graph capture utilities ──────────────────────────────────────── */

/**
 * @brief Opaque CUDA graph handle.
 */
typedef struct vv_dev_graph {
    void* graph;           /**< cudaGraph_t */
    void* graph_exec;      /**< cudaGraphExec_t */
    void* stream;          /**< Capture stream */
    bool  captured;
} vv_dev_graph_t;

/**
 * @brief Begin CUDA graph capture on a stream.
 */
vv_status_t vv_dev_graph_begin_capture(vv_dev_graph_t** graph,
                                         void* stream);

/**
 * @brief End graph capture and instantiate for replay.
 */
vv_status_t vv_dev_graph_end_capture(vv_dev_graph_t* graph);

/**
 * @brief Launch (replay) a captured graph.
 */
vv_status_t vv_dev_graph_launch(vv_dev_graph_t* graph, void* stream);

/**
 * @brief Free CUDA graph.
 */
vv_status_t vv_dev_graph_free(vv_dev_graph_t* graph);

#ifdef __cplusplus
}
#endif

#endif /* VV_PROFILER_H */
