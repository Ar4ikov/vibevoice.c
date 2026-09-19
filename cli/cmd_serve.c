/**
 * @file cmd_serve.c
 * @brief `vv_cli serve` — run the HTTP front end.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/engine.h"
#include "vibevoice/server.h"
#include "vibevoice/kv_quant.h"
#include "vibevoice/tokenizer_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

static vv_server_t* g_server = NULL;

static void on_signal(int sig) {
    (void)sig;
    if (g_server) {
        fprintf(stderr, "\nshutting down...\n");
        vv_server_stop(g_server);
    }
}

static void usage(void) {
    fprintf(stderr,
        "Usage: vv_cli serve --model <dir> [options]\n\n"
        "  --host <addr>         Bind address (default: 0.0.0.0)\n"
        "  --port <n>            Port (default: 8080)\n"
        "  --slots <n>           Concurrent transcriptions (default: 1)\n"
        "  --queue-size <n>      Waiting requests (default: 16; 0 disables)\n"
        "  --max-conns <n>       HTTP connections (default: slots + queue + 8)\n"
        "  --model-name <name>   Name reported by /v1/models\n"
        "  --api-key <key>       Require this bearer token\n"
        "  --gpu <id>            GPU device id (default: 0)\n"
        "  --gpus <list>         Devices to spread slots over: 0,1 | all\n"
        "  --gpu-memory <size>   Cap per device: 80%% | 18GiB | 8192M | bytes,\n"
        "                        or one value per device, comma-separated\n"
        "  --split-mode <how>    auto (default) | replica | layer — a copy of\n"
        "                        the model on each device, or one split across\n"
        "  --max-seq-len <n>     Per-slot KV window (default: 32768)\n"
        "  --kv-cache <fmt>      fp16 | fp8 | fp8-e5m2 | tq4 | tq3 | tq2 | tq1.5\n"
        "  --quant <fmt>         auto (default) | none | nf4 | int4 | int8 —\n"
        "                        weights of a dense checkpoint, quantized at\n"
        "                        load\n"
        "  --acoustic-sampling M mode (default) | fix | gaussian; a draw per\n"
        "                        request, so answers stop being reproducible\n"
        "  --vram-budget <0-1>   Fraction of free VRAM to use\n"
        "  --cpu                 CPU-only\n"
        "  --verbose\n\n"
        "Endpoints: POST /v1/audio/transcriptions, GET /v1/models, "
        "GET /health, GET /metrics\n");
}

int vv_cmd_serve(int argc, char** argv) {
    vv_engine_params_t ep = vv_engine_params_default();
    vv_server_params_t sp = vv_server_params_default();
    bool verbose = false;
    const char* gpus_arg = NULL;
    const char* mem_arg = NULL;

    for (int i = 0; i < argc; i++) {
        const char* a = argv[i];
        const char* next = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (strcmp(a, "--model") == 0 && next) { ep.model_dir = argv[++i]; }
        else if (strcmp(a, "--host") == 0 && next) { sp.host = argv[++i]; }
        else if (strcmp(a, "--port") == 0 && next) { sp.port = atoi(argv[++i]); }
        else if (strcmp(a, "--slots") == 0 && next) { ep.n_slots = atoi(argv[++i]); }
        else if (strcmp(a, "--queue-size") == 0 && next) { sp.queue_size = atoi(argv[++i]); }
        else if (strcmp(a, "--max-conns") == 0 && next) { sp.max_conns = atoi(argv[++i]); }
        else if (strcmp(a, "--model-name") == 0 && next) { sp.model_name = argv[++i]; }
        else if (strcmp(a, "--api-key") == 0 && next) { sp.api_key = argv[++i]; }
        else if (strcmp(a, "--gpu") == 0 && next) { ep.gpu_id = atoi(argv[++i]); }
        else if (strcmp(a, "--gpus") == 0 && next) { gpus_arg = argv[++i]; }
        else if (strcmp(a, "--gpu-memory") == 0 && next) { mem_arg = argv[++i]; }
        else if (strcmp(a, "--split-mode") == 0 && next) {
            const vv_split_mode_t m = vv_split_mode_parse(argv[++i]);
            if (m >= VV_SPLIT_MODE_COUNT) {
                fprintf(stderr, "serve: --split-mode is auto, replica or "
                                "layer, not '%s'\n", argv[i]);
                return 1;
            }
            ep.split_mode = (int)m;
        }
        else if (strcmp(a, "--max-seq-len") == 0 && next) { ep.max_seq_len = atoi(argv[++i]); }
        else if (strcmp(a, "--gpu-layers") == 0 && next) { ep.gpu_layers = atoi(argv[++i]); }
        else if (strcmp(a, "--vram-budget") == 0 && next) { ep.vram_budget = (float)atof(argv[++i]); }
        else if (strcmp(a, "--kv-cache") == 0 && next) {
            vv_kv_format_t f = vv_kv_format_parse(argv[++i]);
            if (f >= VV_KV_FORMAT_COUNT) {
                fprintf(stderr, "error: unknown --kv-cache format '%s'\n", argv[i]);
                return 1;
            }
            ep.kv_format = (int)f;
        }
        else if (strcmp(a, "--quant") == 0 && next) {
            const vv_load_quant_t q = vv_load_quant_parse(argv[++i]);
            if (q >= VV_LOAD_QUANT_COUNT) {
                fprintf(stderr, "error: --quant is auto, none, nf4, int4 or int8, "
                                "not '%s'\n", argv[i]);
                return 1;
            }
            ep.weight_quant = (int)q;
        }
        else if (strcmp(a, "--acoustic-sampling") == 0 && next) {
            vv_acoustic_sampling_t m = vv_acoustic_sampling_parse(argv[++i]);
            if (m >= VV_ACOUSTIC_SAMPLING_COUNT) {
                fprintf(stderr, "error: unknown --acoustic-sampling '%s'"
                                " (mode | fix | gaussian)\n", argv[i]);
                return 1;
            }
            sp.acoustic_sampling = (int)m;
        }
        else if (strcmp(a, "--cpu") == 0) { ep.cpu_only = true; }
        else if (strcmp(a, "--verbose") == 0) { verbose = true; }
        else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage();
            return 0;
        } else {
            fprintf(stderr, "serve: unknown option '%s'\n\n", a);
            usage();
            return 1;
        }
    }

    if (!ep.model_dir) {
        fprintf(stderr, "serve: --model is required\n\n");
        usage();
        return 1;
    }

    if (ep.n_slots < 1 || ep.n_slots > 128 || sp.queue_size < 0 ||
        sp.queue_size > 4096 || sp.max_conns < 0 || sp.max_conns > 8192 ||
        sp.port < 1 || sp.port > 65535) {
        VV_LOG_E("serve: invalid slots, queue size, connection limit or port");
        return 1;
    }
    vv_log_set_level(verbose ? VV_LOG_DEBUG : VV_LOG_INFO);

    if (gpus_arg && vv_gpu_set_parse(gpus_arg, &ep.gpus) != VV_OK) return 1;
    if (mem_arg && vv_gpu_set_caps(mem_arg, &ep.gpus) != VV_OK) return 1;
    if (ep.gpus.n > 0) ep.gpu_id = ep.gpus.id[0];
    if (vv_gpu_set_resolve(&ep.gpus, ep.gpu_id, ep.cpu_only) != VV_OK) return 1;
    if (ep.gpus.n > ep.n_slots && !ep.cpu_only)
        VV_LOG_W("serve: %d devices but only %d slot(s); raise --slots to use "
                 "them all", ep.gpus.n, ep.n_slots);

    vv_engine_t* engine = NULL;
    vv_status_t s = vv_engine_create(&ep, &engine);
    if (s != VV_OK) {
        VV_LOG_E("serve: cannot start engine: %s", vv_status_str(s));
        return 1;
    }

    signal(SIGINT, on_signal);
#ifdef SIGTERM
    signal(SIGTERM, on_signal);
#endif

    /* The model is resident and warm before the first request arrives. */
    VV_LOG_I("serve: ready on http://%s:%d", sp.host, sp.port);
    s = vv_server_run(engine, &sp, &g_server);
    g_server = NULL;

    vv_engine_free(engine);
    return s == VV_OK ? 0 : 1;
}
