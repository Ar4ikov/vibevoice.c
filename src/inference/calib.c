/**
 * @file calib.c
 * @brief SmoothQuant calibration: the dense model transcribes a clip and
 *        every layer records the range of each projection's input.
 *
 * The statistics are the per-channel absmax over every position the model
 * sees, the prompt with the speech features and the transcript it writes,
 * which is where a quantized model spends its time. They are folded into
 * the weights by the loader (smooth.h); the calibration model is freed
 * before the quantized one is loaded, so the two never share memory.
 */

#include "vibevoice/inference.h"
#include "vibevoice/vibevoice.h"
#include "vibevoice/audio.h"
#include "vibevoice/device.h"
#include "vibevoice/quant.h"
#include "vibevoice/smooth.h"

#include <string.h>

vv_status_t vv_smooth_calibrate(const char* model_dir, int gpu_id,
                                const vv_init_params_t* params,
                                const float* pcm, int n_samples,
                                int sample_rate, vv_smooth_stats_t** out) {
    if (!model_dir || !pcm || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (n_samples <= 0 || sample_rate <= 0) return VV_ERR_INVALID_ARG;

    /*
     * One dense context on one device: the statistics live next to the
     * layers that write them, and a layer split would scatter them.
     */
    vv_init_params_t p = params ? *params : vv_init_params_default();
    p.weight_quant = VV_LOAD_QUANT_NONE;
    p.smooth = NULL;
    /* Keep the caller's memory cap for that one device, if it gave one. */
    {
        vv_mem_cap_t cap = { 0, 0.0f };
        for (int i = 0; i < p.gpus.n; i++)
            if (p.gpus.id[i] == gpu_id) cap = p.gpus.cap[i];
        p.gpus.n = 1;
        p.gpus.id[0] = gpu_id;
        p.gpus.cap[0] = cap;
    }
    p.split_mode = VV_SPLIT_REPLICA;
    p.n_slots = 1;
    p.kv_paging = VV_KV_PAGED_OFF;

    const double t0 = vv_time_ms();
    VV_LOG_I("calib: loading the dense model for SmoothQuant statistics");
    vv_inference_ctx_t* ctx = NULL;
    vv_status_t s = vv_inference_init(model_dir, gpu_id, &p, &ctx);
    if (s != VV_OK) {
        VV_LOG_E("calib: cannot load '%s' dense: %s", model_dir,
                 vv_status_str(s));
        return s;
    }

    vv_model_t* m = ctx->model;
    const vv_llm_config_t* c = &m->config.llm;
    vv_smooth_stats_t* st = NULL;
    float* dev = NULL;
    float* audio = NULL;
    int n_audio = 0;
    vv_transcription_t* tr = NULL;

    s = vv_smooth_stats_alloc(m->num_layers, c->hidden_size,
                              c->num_attention_heads * c->head_dim,
                              c->intermediate_size, &st);
    const size_t n = st ? (size_t)st->n_layers * (size_t)st->width : 0;
    if (s == VV_OK && ctx->use_gpu) {
        s = vv_dev_alloc((void**)&dev, n * sizeof(float));
        if (s == VV_OK) s = vv_dev_memset(dev, 0, n * sizeof(float));
    }
    if (s == VV_OK) {
        float* base = dev ? dev : st->absmax;
        for (int i = 0; i < m->num_layers; i++)
            m->layers[i].calib_absmax = base + (size_t)i * st->width;
        s = vv_audio_prepare_ex(pcm, n_samples, sample_rate,
                                ctx->family_ok ? ctx->family.normalize_audio
                                               : true,
                                &audio, &n_audio);
    }
    if (s == VV_OK) {
        vv_inference_params_t ip;
        memset(&ip, 0, sizeof(ip));
        ip.top_k = 1;
        ip.enable_timestamps = true;
        ip.enable_diarize = true;
        s = vv_inference_transcribe(ctx, audio, n_audio, &ip, &tr);
    }
    if (s == VV_OK && dev) {
        s = vv_dev_memcpy_d2h(st->absmax, dev, n * sizeof(float),
                              ctx->compute_stream);
        if (s == VV_OK) s = vv_dev_stream_sync(ctx->compute_stream);
    }
    if (s == VV_OK) {
        const vv_perf_metrics_t* pm = vv_inference_get_perf(ctx);
        st->n_tokens = pm ? (long)pm->prefill_tokens + pm->decode_tokens : 0;
        VV_LOG_I("calib: %.1f s of audio, %ld positions, in %.1f s",
                 (double)n_audio / 24000.0, st->n_tokens,
                 (vv_time_ms() - t0) / 1000.0);
    }

    for (int i = 0; i < m->num_layers; i++) m->layers[i].calib_absmax = NULL;
    if (tr) vv_transcription_free(tr);
    vv_free(audio);
    if (dev) vv_dev_free(dev);
    vv_inference_free(ctx);
    if (s != VV_OK) {
        VV_LOG_E("calib: failed: %s", vv_status_str(s));
        vv_smooth_stats_free(st);
        return s;
    }
    *out = st;
    return VV_OK;
}

vv_status_t vv_smooth_from_args(const char* model_dir, int gpu_id,
                                const vv_init_params_t* params,
                                const char* calib_audio,
                                const char* stats_path,
                                vv_smooth_stats_t** out) {
    if (!out) return VV_ERR_NULL_PTR;
    *out = NULL;
    if (!calib_audio && !stats_path) return VV_OK;
    if (!calib_audio) return vv_smooth_stats_load(stats_path, out);

    float* raw = NULL;
    int n = 0, sr = 0;
    vv_status_t s = vv_audio_load_any(calib_audio, &raw, &n, &sr);
    if (s != VV_OK) {
        VV_LOG_E("calib: cannot read '%s': %s", calib_audio, vv_status_str(s));
        return s;
    }
    s = vv_smooth_calibrate(model_dir, gpu_id, params, raw, n, sr, out);
    vv_free(raw);
    if (s == VV_OK && stats_path) {
        s = vv_smooth_stats_save(*out, stats_path);
        if (s == VV_OK)
            VV_LOG_I("calib: statistics written to '%s'", stats_path);
        else {
            vv_smooth_stats_free(*out);
            *out = NULL;
        }
    }
    return s;
}
