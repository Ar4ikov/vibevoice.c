/**
 * @file cmd_serve.c
 * @brief `vv_cli serve` — run the HTTP front end.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/engine.h"
#include "vibevoice/server.h"
#include "vibevoice/kv_quant.h"
#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/inference.h"
#include "vibevoice/smooth.h"

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
        "  --quant <fmt>         auto (default) | none | nf4 | int4 | int8 |\n"
        "                        w8a8 | w4a8 — weights quantized at load; the\n"
        "                        last two also run on int8 activations\n"
        "  --calib <audio>       SmoothQuant for w8a8 | w4a8: calibrate the\n"
        "                        dense checkpoint on this clip first\n"
        "  --calib-stats <file>  Read the calibration from a file, or, with\n"
        "                        --calib, write it there\n"
        "  --vae <numerics>      auto (default) | float | int8 — speech encoder\n"
        "                        (int8: VibeVoice-ASR-BitNet, VibeASR.cpp's)\n"
        "  --source <from>       BitNet weights: auto | gguf | safetensors\n"
        "  --draft <dir>         DFlash 2 drafter for this model: decode drafts\n"
        "                        a block of tokens and checks it in one pass\n"
        "                        (tools/dflash; same tokens, fewer passes).\n"
        "                        Default: <model>/drafter if there is one;\n"
        "                        none: never\n"
        "  --draft-quant <fmt>   drafter weights: int4 (default) | f16\n"
        "  --draft-block <n>     rows of a block checked per pass (2..block,\n"
        "                        default 4)\n"
        "  --draft-check <how>   exact (default: the transcript is the one\n"
        "                        without --draft, byte for byte) | fast (the\n"
        "                        prefill kernels; a near-tie may go the\n"
        "                        other way)\n"
        "  --head <fmt>          BitNet head on the CPU: auto (exact, int8\n"
        "                        filter) | f16 (full scan) | int8 (approximate)\n"
        "  --attn <backend>      auto (default) | fa1 | fa2 | flashinfer —\n"
        "                        attention kernels; VV_ATTN= does the same\n"
        "  --kv-paged <mode>     auto (default: on with several slots) | on |\n"
        "                        off — slots share one pool of KV pages\n"
        "  --acoustic-sampling M mode (default) | fix | gaussian; a draw per\n"
        "                        request, so answers stop being reproducible\n"
        "  --stream-reserve <s>  Streaming models: seconds of audio whose KV a\n"
        "                        live session must get at open, or it is\n"
        "                        refused (503 / close 1013). Default 180;\n"
        "                        0 = none\n"
        "  --stream-idle <s>     Close a WebSocket after this long without\n"
        "                        audio or a message; pings do not count\n"
        "                        (default: 60)\n"
        "  --stream-slot-wait <s> How long a new live session waits for a\n"
        "                        free slot before server_overloaded\n"
        "                        (default: 5)\n"
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
    const char* calib = NULL;
    const char* calib_stats = NULL;

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
        else if (strcmp(a, "--attn") == 0 && next) {
            const vv_attn_backend_t b = vv_attn_backend_parse(argv[++i]);
            if (b >= VV_ATTN_BACKEND_COUNT) {
                fprintf(stderr, "serve: --attn is auto, fa1, fa2 or "
                                "flashinfer, not '%s'\n", argv[i]);
                return 1;
            }
            ep.attn_backend = (int)b;
        }
        else if (strcmp(a, "--kv-paged") == 0 && next) {
            const int m = vv_kv_paging_parse(argv[++i]);
            if (m < 0) {
                fprintf(stderr, "serve: --kv-paged is auto, on or off, not "
                                "'%s'\n", argv[i]);
                return 1;
            }
            ep.kv_paging = m;
        }
        else if (strcmp(a, "--kv-cache") == 0 && next) {
            vv_kv_format_t f = vv_kv_format_parse(argv[++i]);
            if (f >= VV_KV_FORMAT_COUNT) {
                fprintf(stderr, "error: unknown --kv-cache format '%s'\n", argv[i]);
                return 1;
            }
            ep.kv_format = (int)f;
        }
        else if (strcmp(a, "--calib") == 0 && next) { calib = argv[++i]; }
        else if (strcmp(a, "--calib-stats") == 0 && next) {
            calib_stats = argv[++i];
        }
        else if (strcmp(a, "--quant") == 0 && next) {
            const vv_load_quant_t q = vv_load_quant_parse(argv[++i]);
            if (q >= VV_LOAD_QUANT_COUNT) {
                fprintf(stderr, "error: --quant is auto, none, nf4, int4, int8, w8a8 or w4a8, "
                                "not '%s'\n", argv[i]);
                return 1;
            }
            ep.weight_quant = (int)q;
        }
        else if (strcmp(a, "--vae") == 0 && next) {
            const vv_vae_numerics_t v = vv_vae_numerics_parse(argv[++i]);
            if (v >= VV_VAE_COUNT) {
                fprintf(stderr, "error: --vae is auto, float or int8, not '%s'\n",
                        argv[i]);
                return 1;
            }
            ep.vae_numerics = (int)v;
        }
        else if (strcmp(a, "--source") == 0 && next) {
            const vv_weights_source_t v = vv_weights_source_parse(argv[++i]);
            if (v >= VV_SOURCE_COUNT) {
                fprintf(stderr, "error: --source is auto, gguf or safetensors, "
                                "not '%s'\n", argv[i]);
                return 1;
            }
            ep.weights_source = (int)v;
        }
        else if (strcmp(a, "--draft") == 0 && next) {
            /* "none": an empty dir, which also skips <model>/drafter. */
            ep.draft_dir = strcmp(argv[++i], "none") == 0 ? "" : argv[i];
        }
        else if (strcmp(a, "--draft-quant") == 0 && next) {
            const vv_drafter_quant_t v = vv_drafter_quant_parse(argv[++i]);
            if (v >= VV_DRAFTER_QUANT_COUNT) {
                fprintf(stderr, "error: --draft-quant is int4 or f16, not "
                                "'%s'\n", argv[i]);
                return 1;
            }
            ep.draft_quant = (int)v;
        }
        else if (strcmp(a, "--draft-check") == 0 && next) {
            const vv_draft_check_t v = vv_draft_check_parse(argv[++i]);
            if (v >= VV_DRAFT_CHECK_COUNT) {
                fprintf(stderr, "error: --draft-check is exact or fast, not "
                                "'%s'\n", argv[i]);
                return 1;
            }
            ep.draft_check = (int)v;
        }
        else if (strcmp(a, "--draft-block") == 0 && next) {
            ep.draft_block = atoi(argv[++i]);
            if (ep.draft_block < 2) {
                fprintf(stderr, "error: --draft-block is 2 or more, not '%s'\n",
                        argv[i]);
                return 1;
            }
        }
        else if (strcmp(a, "--head") == 0 && next) {
            const vv_head_format_t v = vv_head_format_parse(argv[++i]);
            if (v >= VV_HEAD_COUNT) {
                fprintf(stderr, "error: --head is auto, f16 or int8, not '%s'\n",
                        argv[i]);
                return 1;
            }
            ep.head_format = (int)v;
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
        else if (strcmp(a, "--stream-reserve") == 0 && next) {
            sp.stream_reserve_sec = atof(argv[++i]);
            if (sp.stream_reserve_sec < 0) sp.stream_reserve_sec = 0;
        }
        else if (strcmp(a, "--stream-idle") == 0 && next) {
            const double v = atof(argv[++i]);
            if (!(v > 0 && v < 86400)) {
                fprintf(stderr, "serve: --stream-idle is seconds, 0..86400\n");
                return 1;
            }
            sp.stream_idle_ms = (int)(v * 1000.0);
        }
        else if (strcmp(a, "--stream-slot-wait") == 0 && next) {
            const double v = atof(argv[++i]);
            if (!(v >= 0 && v < 3600)) {
                fprintf(stderr, "serve: --stream-slot-wait is seconds, "
                                "0..3600\n");
                return 1;
            }
            sp.stream_slot_wait_ms = (int)(v * 1000.0);
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

    if (vv_gpu_set_from_flags(gpus_arg, mem_arg, ep.cpu_only, &ep.gpu_id,
                              &ep.gpus) != VV_OK)
        return 1;
    /*
     * Fewer slots than devices idles a device only when replicas were asked
     * for by name: `auto` then splits the layers, which uses every device
     * (engine.c), and `layer` always does.
     */
    if (ep.gpus.n > ep.n_slots && !ep.cpu_only &&
        ep.split_mode == VV_SPLIT_REPLICA)
        VV_LOG_W("serve: --split-mode replica with %d devices but %d slot(s) "
                 "leaves %d of them idle; raise --slots, or use --split-mode "
                 "layer", ep.gpus.n, ep.n_slots, ep.gpus.n - ep.n_slots);

    {
        const char* sv = getenv("VV_SAVE_TOKENS");
        if (sv && sv[0] && ep.n_slots > 1)
            VV_LOG_W("serve: VV_SAVE_TOKENS is a single-request debugging "
                     "aid; with %d slots concurrent requests overwrite '%s'",
                     ep.n_slots, sv);
    }

    vv_smooth_stats_t* smooth = NULL;
    if (calib || calib_stats) {
        if (!vv_load_quant_takes_smooth(ep.weight_quant)) {
            fprintf(stderr, "serve: --calib/--calib-stats fold into weights "
                            "quantized at load; add --quant w8a8, w4a8, "
                            "int8, int4 or nf4\n");
            return 1;
        }
        /* The dense calibration load obeys the same placement limits. */
        vv_init_params_t ip = vv_init_params_default();
        ip.cpu_only = ep.cpu_only;
        ip.max_seq_len = ep.max_seq_len;
        ip.gpu_layers = ep.gpu_layers;
        ip.vram_budget = ep.vram_budget;
        ip.attn_backend = ep.attn_backend;
        ip.gpus = ep.gpus;
        if (vv_smooth_from_args(ep.model_dir, ep.gpu_id, &ip, calib,
                                calib_stats, &smooth) != VV_OK)
            return 1;
        ep.smooth = smooth;
    }

    vv_engine_t* engine = NULL;
    vv_status_t s = vv_engine_create(&ep, &engine);
    vv_smooth_stats_free(smooth);
    ep.smooth = NULL;
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
