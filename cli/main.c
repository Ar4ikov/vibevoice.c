/**
 * @file main.c
 * @brief CLI executable for vibevoice.c — VibeVoice-ASR inference.
 *
 * Usage:
 *   vv_cli --model <path> --audio <file.wav> [--output <file.json>]
 *          [--gpu <id>] [--max-tokens <N>] [--hotwords "word1,word2"]
 *          [--kv-cache FMT] [--verbose]
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/stream.h"
#include "vibevoice/tokenizer_encoder.h"
#include "vibevoice/kv_quant.h"
#include "vibevoice/audio.h"
#include "vibevoice/inference.h"
#include "vibevoice/smooth.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <time.h>
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif

typedef struct {
    const char* model_dir;
    const char* audio_path;
    const char* output_path;
    /* --trt-acoustic / --trt-semantic: accepted for one release, ignored. */
    bool        trt_flag_seen;
    int         gpu_id;
    const char* gpus;
    const char* gpu_memory;
    const char* split_mode;
    int         max_tokens;
    int         max_seq_len;
    int         gpu_layers;
    const char* kv_cache;
    const char* quant;
    const char* calib;
    const char* calib_stats;
    const char* attn;
    const char* source;
    const char* vae;
    const char* head;
    const char* draft;
    const char* kv_paged;
    bool        cpu_only;
    float       vram_budget;
    bool        verbose;
    const char* hotwords;
    const char* acoustic_sampling;
    unsigned long long acoustic_seed;
} cli_args_t;


/**
 * @brief Split "a, b ,c" into a counted array of trimmed strings.
 *
 * The pieces point into @p scratch, which the caller keeps alive for as long
 * as the array is used.
 */
static int split_hotwords(const char* csv, char* scratch, size_t scratch_size,
                          const char** out, int max_out) {
    if (!csv || !csv[0]) return 0;
    snprintf(scratch, scratch_size, "%s", csv);

    int n = 0;
    char* p = scratch;
    while (*p && n < max_out) {
        while (*p == ' ' || *p == '\t') p++;
        char* start = p;
        char* comma = strchr(p, ',');
        if (comma) { *comma = '\0'; p = comma + 1; }
        else       { p += strlen(p); }
        size_t len = strlen(start);
        while (len > 0 && (start[len - 1] == ' ' || start[len - 1] == '\t'))
            start[--len] = '\0';
        if (len > 0) out[n++] = start;
    }
    return n;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "vibevoice.c v%s — Pure C VibeVoice-ASR runtime\n\n"
        "Usage: %s --model <dir> --audio <wav> [options]\n\n"
        "Required:\n"
        "  --model <dir>         Model directory (safetensors + config.json)\n"
        "  --audio <file>        Input WAV audio file\n\n"
        "Optional:\n"
        "  --output <file>       Output JSON file (default: stdout)\n"
        "  --gpu <id>            GPU device ID (default: 0)\n"
        "  --gpus <list>         Devices to use: 0,1 | all (default: --gpu)\n"
        "  --gpu-memory <size>   Cap per device: 80%% | 18GiB | 8192M | bytes,\n"
        "                        or one value per device, comma-separated\n"
        "  --split-mode <how>    auto (default) | layer — how to use several\n"
        "                        devices; a single file can only use layer\n"
        "  --max-tokens <N>      Max decode tokens (default: 64000)\n"
        "  --max-seq-len <N>     KV-cache window in tokens (default: 32768)\n"
        "  --hotwords <words>    Comma-separated hotwords\n"
        "  --kv-cache FMT        KV-cache storage: fp16 (default), fp8,\n"
        "                        fp8-e5m2, tq4, tq3, tq2, tq1.5\n"
        "  --quant <fmt>         Weights: auto (default: as the checkpoint\n"
        "                        stores them) | none | nf4 | int4 | int8 —\n"
        "                        quantize a dense checkpoint at load — or\n"
        "                        w8a8 | w4a8: int8 / int4 weights on int8\n"
        "                        per-token activations (dense, AWQ/GPTQ for\n"
        "                        w4a8, or compressed-tensors checkpoints)\n"
        "  --calib <audio>       SmoothQuant: run the dense checkpoint on this\n"
        "                        clip first and fold its activation ranges\n"
        "                        into the weights --quant quantizes\n"
        "  --calib-stats <file>  Read those ranges from a file instead, or,\n"
        "                        with --calib, write them there\n"
        "  --vae <numerics>      Speech encoder: auto (default) | float |\n"
        "                        int8 (VibeVoice-ASR-BitNet: VibeASR.cpp's\n"
        "                        int8 encoder, bit-exact to it; the CPU\n"
        "                        default)\n"
        "  --source <from>       BitNet weights: auto (default: the GGUF pair\n"
        "                        when present) | gguf | safetensors\n"
        "  --draft <dir>         DFlash 2 drafter for this model: decode drafts\n"
        "                        a block of tokens and checks it in one pass\n"
        "                        (tools/dflash; same tokens, fewer passes)\n"
        "  --head <fmt>          BitNet LM head on the CPU: auto (default: the\n"
        "                        F16 head's exact argmax through an int8\n"
        "                        filter) | f16 (full scan) | int8 (int8 rows\n"
        "                        only, approximate)\n"
        "  --attn <backend>      auto (default) | fa1 | fa2 | flashinfer —\n"
        "                        attention kernels; VV_ATTN= does the same\n"
        "  --kv-paged <mode>     auto (default) | on | off — take KV from a\n"
        "                        pool of 64-position pages\n"
        "  --vram-budget <0-1>   VRAM fraction for model (default: 1.0)\n"
        "  --gpu-layers <N>      Layers to keep on the GPU (-1 = fit to VRAM)\n"
        "  --acoustic-sampling M mode (default, deterministic) | fix |\n"
        "                        gaussian (what the checkpoint asks for)\n"
        "  --seed <N>            Seed for that draw (default: from the clock)\n"
        "  --cpu                 CPU-only mode (no GPU)\n"
        "  --verbose             Enable debug logging\n"
        "  --version             Print version and revision, then exit\n"
        "  --help                Show this message\n\n"
        "Commands:\n"
        "  serve                 Run the OpenAI-compatible HTTP server\n"
        "  chat                  Interactive prompt, model stays loaded\n"
        "  mic                   Live microphone transcription\n"
        "  devices               List the capture devices mic/chat can use\n"
        "  (none)                Transcribe one file and exit\n\n"
        "Run `%s <command> --help` for that command's own options.\n",
        vv_version(), prog, prog);
}

static int parse_args(int argc, char** argv, cli_args_t* args) {
    memset(args, 0, sizeof(*args));
    args->max_tokens = 64000;
    args->vram_budget = 1.0f;
    args->gpu_layers = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args->model_dir = argv[++i];
        } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            args->audio_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args->output_path = argv[++i];
        } else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            args->gpu_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpus") == 0 && i + 1 < argc) {
            args->gpus = argv[++i];
        } else if (strcmp(argv[i], "--gpu-memory") == 0 && i + 1 < argc) {
            args->gpu_memory = argv[++i];
        } else if (strcmp(argv[i], "--split-mode") == 0 && i + 1 < argc) {
            args->split_mode = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            args->max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-seq-len") == 0 && i + 1 < argc) {
            args->max_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hotwords") == 0 && i + 1 < argc) {
            args->hotwords = argv[++i];
        } else if ((strcmp(argv[i], "--trt-acoustic") == 0 ||
                    strcmp(argv[i], "--trt-semantic") == 0) && i + 1 < argc) {
            /* The TensorRT path never got past stubs and is gone; the flags
               stay parseable for one release so existing scripts still run. */
            args->trt_flag_seen = true;
            ++i;
        } else if (strcmp(argv[i], "--attn") == 0 && i + 1 < argc) {
            args->attn = argv[++i];
        } else if (strcmp(argv[i], "--kv-paged") == 0 && i + 1 < argc) {
            args->kv_paged = argv[++i];
        } else if (strcmp(argv[i], "--kv-cache") == 0 && i + 1 < argc) {
            args->kv_cache = argv[++i];
        } else if (strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            args->quant = argv[++i];
        } else if (strcmp(argv[i], "--calib") == 0 && i + 1 < argc) {
            args->calib = argv[++i];
        } else if (strcmp(argv[i], "--calib-stats") == 0 && i + 1 < argc) {
            args->calib_stats = argv[++i];
        } else if (strcmp(argv[i], "--vae") == 0 && i + 1 < argc) {
            args->vae = argv[++i];
        } else if (strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
            args->source = argv[++i];
        } else if (strcmp(argv[i], "--head") == 0 && i + 1 < argc) {
            args->head = argv[++i];
        } else if (strcmp(argv[i], "--draft") == 0 && i + 1 < argc) {
            args->draft = argv[++i];
        } else if (strcmp(argv[i], "--vram-budget") == 0 && i + 1 < argc) {
            args->vram_budget = (float)atof(argv[++i]);
            if (args->vram_budget < 0.0f) args->vram_budget = 0.0f;
            if (args->vram_budget > 1.0f) args->vram_budget = 1.0f;
        } else if (strcmp(argv[i], "--gpu-layers") == 0 && i + 1 < argc) {
            args->gpu_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--acoustic-sampling") == 0 && i + 1 < argc) {
            args->acoustic_sampling = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args->acoustic_seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--cpu") == 0) {
            args->cpu_only = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            args->verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            return -1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return -1;
        }
    }

    if (!args->model_dir || !args->audio_path) {
        fprintf(stderr, "Error: --model and --audio are required\n\n");
        return -1;
    }

    return 0;
}

int vv_cmd_serve(int argc, char** argv);
int vv_cmd_chat(int argc, char** argv);
int vv_cmd_mic(int argc, char** argv);
int vv_cmd_devices(int argc, char** argv);

/* ─── Live output for the streaming model ─────────────────────────────── */

typedef struct {
    double* ms;       /* per chunk: prefill + decode */
    double* enc_ms;   /* per chunk: encode (VV_STREAM_PROFILE=1) */
    int     n, cap;
} cli_stream_t;

static void cli_stream_event(void* user, const vv_stream_event_t* ev) {
    cli_stream_t* c = (cli_stream_t*)user;
    if (ev->type == VV_STREAM_EVENT_DELTA) {
        fwrite(ev->text, 1, ev->text_len, stdout);
        fflush(stdout);
    } else if (ev->type == VV_STREAM_EVENT_CHUNK) {
        if (c->n == c->cap) {
            const int nc = c->cap ? c->cap * 2 : 64;
            double* a = (double*)vv_realloc(c->ms, sizeof(double) * (size_t)nc);
            if (!a) return;
            c->ms = a;
            double* b = (double*)vv_realloc(c->enc_ms,
                                            sizeof(double) * (size_t)nc);
            if (!b) return;
            c->enc_ms = b;
            c->cap = nc;
        }
        c->ms[c->n] = ev->prefill_ms + ev->decode_ms;
        c->enc_ms[c->n] = ev->encode_ms;
        c->n++;
        VV_LOG_D("chunk %lld [%.2f-%.2f s]: %d tokens, prefill %.1f ms "
                 "(encode %.1f), decode %.1f ms, KV %lld",
                 (long long)ev->chunk_index, ev->audio_start, ev->audio_end,
                 ev->n_tokens, ev->prefill_ms, ev->encode_ms, ev->decode_ms,
                 (long long)ev->kv_len);
    } else if (ev->type == VV_STREAM_EVENT_ERROR) {
        VV_LOG_E("stream: chunk %lld failed: %s", (long long)ev->chunk_index,
                 vv_status_str(ev->status));
    }
}

static int cmp_double(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    /* Enable UTF-8 console output */
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

    /*
     * Answered before anything else is parsed, so that the tag on a
     * container image can be checked against the binary it holds.
     */
    if (argc > 1 && (strcmp(argv[1], "--version") == 0 ||
                     strcmp(argv[1], "-V") == 0)) {
        printf("vibevoice.c %s (%s) [%s]\n",
               vv_version(), vv_build_ref(), vv_build_features());
        return 0;
    }

    /*
     * Subcommands. Without one the arguments are read as a single
     * transcription, which is how the CLI has always behaved.
     */
    if (argc > 1 && argv[1][0] != '-') {
        if (strcmp(argv[1], "serve") == 0)
            return vv_cmd_serve(argc - 2, argv + 2);
        if (strcmp(argv[1], "chat") == 0)
            return vv_cmd_chat(argc - 2, argv + 2);
        if (strcmp(argv[1], "mic") == 0)
            return vv_cmd_mic(argc - 2, argv + 2);
        if (strcmp(argv[1], "devices") == 0)
            return vv_cmd_devices(argc - 2, argv + 2);
        if (strcmp(argv[1], "transcribe") == 0) { argc--; argv++; }
        else {
            fprintf(stderr, "unknown command '%s'\n\n", argv[1]);
            print_usage(argv[0]);
            return 1;
        }
    }

    cli_args_t args;
    if (parse_args(argc, argv, &args) != 0) {
        print_usage(argv[0]);
        return 1;
    }

    /* Configure logging */
    vv_log_set_level(args.verbose ? VV_LOG_DEBUG : VV_LOG_INFO);

    VV_LOG_I("vibevoice.c %s (%s)", vv_version(), vv_build_ref());
    if (args.trt_flag_seen)
        VV_LOG_W("--trt-acoustic/--trt-semantic are deprecated and ignored: "
                 "the speech encoder runs on its own CUDA kernels. The flags "
                 "go away in the next release.");
    VV_LOG_I("Model: %s", args.model_dir);
    VV_LOG_I("Audio: %s", args.audio_path);

    vv_gpu_set_t gpus;
    memset(&gpus, 0, sizeof(gpus));
    if (vv_gpu_set_from_flags(args.gpus, args.gpu_memory, args.cpu_only,
                              &args.gpu_id, &gpus) != VV_OK)
        return 1;
    /*
     * One file is one request, and a request cannot be in two places at
     * once: replicas would leave every device but the first idle, so the
     * only thing a second card can do here is hold part of the model.
     */
    vv_split_mode_t split = VV_SPLIT_AUTO;
    if (args.split_mode) {
        split = vv_split_mode_parse(args.split_mode);
        if (split >= VV_SPLIT_MODE_COUNT) {
            VV_LOG_E("split-mode: '%s' is not auto, replica or layer",
                     args.split_mode);
            return 1;
        }
    }
    if (gpus.n > 1 && !args.cpu_only) {
        if (split == VV_SPLIT_AUTO) split = VV_SPLIT_LAYER;
        if (split == VV_SPLIT_REPLICA)
            VV_LOG_W("gpus: replicas do nothing for a single file; gpu %d "
                     "will do the work", gpus.id[0]);
    }

    double t_start = get_time_ms();

    /* Step 1: Load and preprocess audio */
    double t0 = get_time_ms();
    float* audio = NULL;
    int n_samples = 0;

    /*
     * Decoded now, prepared once the model is up: whether it is loudness
     * normalized is the model's call (the streaming checkpoint says no).
     */
    float* raw_audio = NULL;
    int raw_len = 0, raw_sr = 0;
    vv_status_t s = vv_audio_load_any(args.audio_path, &raw_audio, &raw_len,
                                      &raw_sr);
    if (s != VV_OK) {
        VV_LOG_E("Failed to load audio: %s", vv_status_str(s));
        return 1;
    }
    double t1 = get_time_ms();
    VV_LOG_I("Audio loaded: %d samples at %d Hz in %.1f ms",
             raw_len, raw_sr, t1 - t0);

    /* Step 2: Initialize inference */
    t0 = get_time_ms();
    vv_inference_ctx_t* ctx = NULL;
    vv_init_params_t init_params = vv_init_params_default();
    if (args.quant) {
        const vv_load_quant_t q = vv_load_quant_parse(args.quant);
        if (q >= VV_LOAD_QUANT_COUNT) {
            fprintf(stderr, "error: --quant is auto, none, nf4, int4, int8, w8a8 or w4a8, "
                            "not '%s'\n", args.quant);
            vv_free(raw_audio);
            return 1;
        }
        init_params.weight_quant = (int)q;
    }
    if (args.vae) {
        const vv_vae_numerics_t v = vv_vae_numerics_parse(args.vae);
        if (v >= VV_VAE_COUNT) {
            fprintf(stderr, "error: --vae is auto, float or int8, not '%s'\n",
                    args.vae);
            vv_free(raw_audio);
            return 1;
        }
        init_params.vae_numerics = (int)v;
    }
    if (args.source) {
        const vv_weights_source_t v = vv_weights_source_parse(args.source);
        if (v >= VV_SOURCE_COUNT) {
            fprintf(stderr, "error: --source is auto, gguf or safetensors, "
                            "not '%s'\n", args.source);
            vv_free(raw_audio);
            return 1;
        }
        init_params.weights_source = (int)v;
    }
    if (args.head) {
        const vv_head_format_t v = vv_head_format_parse(args.head);
        if (v >= VV_HEAD_COUNT) {
            fprintf(stderr, "error: --head is auto, f16 or int8, not '%s'\n",
                    args.head);
            vv_free(raw_audio);
            return 1;
        }
        init_params.head_format = (int)v;
    }
    init_params.draft_dir = args.draft;
    if (args.kv_cache) {
        vv_kv_format_t f = vv_kv_format_parse(args.kv_cache);
        if (f >= VV_KV_FORMAT_COUNT) {
            fprintf(stderr, "error: unknown --kv-cache format '%s'\n",
                    args.kv_cache);
            return 1;
        }
        init_params.kv_format = (int)f;
    }
    if (args.attn) {
        const vv_attn_backend_t b = vv_attn_backend_parse(args.attn);
        if (b >= VV_ATTN_BACKEND_COUNT) {
            fprintf(stderr, "error: --attn is auto, fa1, fa2 or flashinfer, "
                            "not '%s'\n", args.attn);
            return 1;
        }
        init_params.attn_backend = (int)b;
    }
    if (args.kv_paged) {
        const int m = vv_kv_paging_parse(args.kv_paged);
        if (m < 0) {
            fprintf(stderr, "error: --kv-paged is auto, on or off, not '%s'\n",
                    args.kv_paged);
            return 1;
        }
        init_params.kv_paging = m;
    }
    init_params.vram_budget = args.vram_budget;
    init_params.cpu_only = args.cpu_only;
    if (args.max_seq_len > 0) init_params.max_seq_len = args.max_seq_len;
    init_params.gpu_layers = args.gpu_layers;
    init_params.gpus = gpus;
    init_params.split_mode = (int)split;
    vv_smooth_stats_t* smooth = NULL;
    if ((args.calib || args.calib_stats) &&
        !vv_load_quant_takes_smooth(init_params.weight_quant)) {
        fprintf(stderr, "error: --calib/--calib-stats fold into weights "
                        "quantized at load; add --quant w8a8, w4a8, int8, "
                        "int4 or nf4\n");
        vv_free(raw_audio);
        return 1;
    }
    s = vv_smooth_from_args(args.model_dir, args.gpu_id, &init_params,
                            args.calib, args.calib_stats, &smooth);
    if (s != VV_OK) {
        VV_LOG_E("SmoothQuant calibration failed: %s", vv_status_str(s));
        vv_free(raw_audio);
        return 1;
    }
    init_params.smooth = smooth;
    s = vv_inference_init(args.model_dir, args.gpu_id, &init_params, &ctx);
    vv_smooth_stats_free(smooth);
    if (s != VV_OK) {
        VV_LOG_E("Failed to initialize inference: %s", vv_status_str(s));
        vv_free(raw_audio);
        return 1;
    }
    t1 = get_time_ms();
    VV_LOG_I("Model loaded in %.1f ms", t1 - t0);

    /* Resample to 24 kHz; normalize to -25 dBFS unless the model says not. */
    if (ctx->family_ok && ctx->family.vibeasr_audio)
        s = vv_audio_prepare_vibeasr(raw_audio, raw_len, raw_sr,
                                     ctx->family.normalize_audio,
                                     &audio, &n_samples);
    else
        s = vv_audio_prepare_ex(raw_audio, raw_len, raw_sr,
                                ctx->family_ok ? ctx->family.normalize_audio
                                               : true,
                                &audio, &n_samples);
    vv_free(raw_audio);
    if (s != VV_OK) {
        VV_LOG_E("Failed to prepare audio: %s", vv_status_str(s));
        vv_inference_free(ctx);
        return 1;
    }
    VV_LOG_I("Audio prepared: %d samples (%.2f sec) at 24 kHz",
             n_samples, (float)n_samples / 24000.0f);

    /* Step 3: Transcribe */
    t0 = get_time_ms();
    vv_inference_params_t params;
    memset(&params, 0, sizeof(params));
    params.max_new_tokens = args.max_tokens;
    params.temperature = 0.0f;
    params.top_k = 1;
    params.enable_timestamps = true;
    params.enable_diarize = true;
    params.acoustic_seed = args.acoustic_seed;
    if (args.acoustic_sampling) {
        vv_acoustic_sampling_t m =
            vv_acoustic_sampling_parse(args.acoustic_sampling);
        if (m >= VV_ACOUSTIC_SAMPLING_COUNT) {
            fprintf(stderr, "error: unknown --acoustic-sampling '%s'"
                            " (mode | fix | gaussian)\n",
                    args.acoustic_sampling);
            return 1;
        }
        params.acoustic_sampling = (int)m;
    }

    char hotword_buf[512];
    const char* hotword_list[32];
    params.num_hotwords = split_hotwords(args.hotwords, hotword_buf,
                                          sizeof(hotword_buf),
                                          hotword_list, 32);
    params.hotwords = params.num_hotwords ? hotword_list : NULL;
    if (params.num_hotwords)
        VV_LOG_I("Hotwords: %d term(s)", params.num_hotwords);

    vv_transcription_t* result = NULL;
    const bool chunked = ctx->family_ok &&
                         ctx->family.mode == VV_GEN_CHUNKED;
    cli_stream_t live;
    memset(&live, 0, sizeof(live));
    if (chunked) {
        /*
         * The streaming model writes text chunk by chunk; show it as it
         * comes, then the transcript and the segments as usual.
         */
        char ctx_info[VV_HOTWORDS_MAX];
        vv_hotwords_join(params.hotwords, params.num_hotwords, ctx_info,
                         sizeof(ctx_info));
        fprintf(stderr, "\n--- streaming (%d+%d frames per chunk) ---\n",
                ctx->family.chunk_frames, ctx->family.lookahead_frames);
        s = vv_stream_transcribe(ctx, audio, n_samples,
                                 ctx_info[0] ? ctx_info : NULL,
                                 cli_stream_event, &live, &result);
        fprintf(stderr, "\n--- end stream (%d chunks) ---\n", live.n);
    } else {
        s = vv_inference_transcribe(ctx, audio, n_samples, &params, &result);
    }
    if (s != VV_OK) {
        VV_LOG_E("Transcription failed: %s", vv_status_str(s));
        vv_inference_free(ctx);
        vv_free(audio);
        return 1;
    }
    t1 = get_time_ms();

    double t_total = get_time_ms() - t_start;
    float audio_duration = (float)n_samples / 24000.0f;

    /* Step 4: Output result */
    if (result && result->full_text) {
        if (args.output_path) {
            char* json = NULL;
            extern vv_status_t vv_transcription_to_json(
                const vv_transcription_t* tr, char** json_str);
            vv_transcription_to_json(result, &json);
            if (json) {
                FILE* f = fopen(args.output_path, "w");
                if (f) {
                    fputs(json, f);
                    fclose(f);
                    VV_LOG_I("Output written to %s", args.output_path);
                }
                free(json);
            }
        } else {
            printf("\n=== Transcription ===\n%s\n", result->full_text);

            if (result->num_segments > 0) {
                printf("\n=== Segments ===\n");
                for (int i = 0; i < result->num_segments; i++) {
                    printf("[%.2f - %.2f] %s: %s\n",
                           result->segments[i].start_time,
                           result->segments[i].end_time,
                           result->segments[i].speaker ?
                               result->segments[i].speaker : "?",
                           result->segments[i].text ?
                               result->segments[i].text : "");
                }
            }
        }
    }

    /* ── Performance dashboard ── */
    const vv_perf_metrics_t* perf = vv_inference_get_perf(ctx);
    if (perf) {
        printf("\n");
        printf("=========================================="
               "==========================================\n");
        printf("  PERFORMANCE METRICS\n");
        printf("=========================================="
               "==========================================\n");

        /* Timing breakdown */
        printf("\n  Timing breakdown:\n");
        printf("    Audio encoding       %8.1f ms\n",
               perf->audio_encode_ms);
        printf("    Sequence build       %8.1f ms\n",
               perf->sequence_build_ms);
        printf("    LLM Prefill          %8.1f ms   "
               "(%d tokens, %d layers)\n",
               perf->prefill_ms, perf->prefill_tokens, perf->num_layers);
        printf("    Autoregressive decode%8.1f ms   (%d tokens)\n",
               perf->decode_ms, perf->decode_tokens);
        if (perf->decode_layers_ms > 0.0 && perf->decode_tokens > 0) {
            printf("      layers             %8.1f ms   (%.2f ms/token)\n",
                   perf->decode_layers_ms,
                   perf->decode_layers_ms / perf->decode_tokens);
            printf("      lm_head + sample   %8.1f ms   (%.2f ms/token)\n",
                   perf->decode_head_ms,
                   perf->decode_head_ms / perf->decode_tokens);
            printf("      embed              %8.1f ms\n",
                   perf->decode_embed_ms);
        }
        printf("    Post-processing      %8.1f ms\n",
               perf->postprocess_ms);
        printf("    ----------------------------------------\n");
        printf("    Total inference      %8.1f ms\n",
               perf->total_ms);
        printf("    Total wall-clock     %8.1f ms   (incl. model load)\n",
               t_total);

        /* Key metrics */
        printf("\n  Key metrics:\n");
        printf("    TTFT                 %8.1f ms\n",
               perf->ttft_ms);
        printf("    Prefill speed        %8.0f tok/s\n",
               perf->prefill_tok_per_sec);
        printf("    Decode speed         %8.1f tok/s\n",
               perf->decode_tok_per_sec);
        printf("    RTF                  %8.3f       "
               "(%.2fs audio / %.2fs compute)\n",
               perf->rtf, perf->audio_duration_sec,
               perf->total_ms / 1000.0);

        /* Tokens */
        if (chunked && live.n > 0) {
            /* Per-chunk latency: what a live caller waits after a window's
             * last sample arrives before its text is complete. */
            double sum = 0.0, enc = 0.0;
            for (int i = 0; i < live.n; i++) sum += live.ms[i];
            for (int i = 0; i < live.n; i++) enc += live.enc_ms[i];
            qsort(live.ms, (size_t)live.n, sizeof(double), cmp_double);
            printf("\n  Streaming (%d chunks of %.2f s):\n", live.n,
                   (double)ctx->family.chunk_frames *
                   ctx->family.frame_samples / ctx->family.sample_rate);
            printf("    Chunk latency        %8.1f ms mean, p50 %.1f, p95 %.1f, "
                   "max %.1f\n", sum / live.n, live.ms[live.n / 2],
                   live.ms[(live.n * 95) / 100 < live.n
                           ? (live.n * 95) / 100 : live.n - 1],
                   live.ms[live.n - 1]);
            if (enc > 0.0)
                printf("      of which encode    %8.1f ms mean\n", enc / live.n);
            printf("    Prompt prefill       %8.1f ms\n", perf->ttft_ms);
        }

        printf("\n  Token counts:\n");
        printf("    Prefill tokens       %8d\n", perf->prefill_tokens);
        printf("    Audio frames         %8d\n", perf->audio_frames);
        printf("    Generated tokens     %8d\n", perf->decode_tokens);
        printf("    Total tokens         %8d\n",
               perf->prefill_tokens + perf->decode_tokens);

        /* KV cache */
        printf("\n  KV cache:\n");
        printf("    Positions used       %8d / %d (%.1f%%)\n",
               perf->kv_cache_used, perf->kv_cache_max,
               perf->kv_cache_pct);
        printf("    Format               %8s\n",
               vv_kv_format_name((vv_kv_format_t)perf->kv_format));
        printf("    Attention            %8s\n",
               vv_attn_backend_name((vv_attn_backend_t)perf->attn_backend));

        /* GPU memory */
        if (perf->vram_total_bytes > 0) {
            printf("\n  GPU memory:\n");
            printf("    VRAM used            %8.1f MB / %.1f MB (%.1f%%)\n",
                   (double)perf->vram_used_bytes / (1024.0 * 1024.0),
                   (double)perf->vram_total_bytes / (1024.0 * 1024.0),
                   100.0 * (double)perf->vram_used_bytes /
                       (double)perf->vram_total_bytes);
            printf("    VRAM free            %8.1f MB\n",
                   (double)perf->vram_free_bytes / (1024.0 * 1024.0));
            printf("    Workspace            %8zu MB\n",
                   perf->workspace_mb);
        }

        /* Audio info */
        printf("\n  Audio:\n");
        printf("    Duration             %8.2f sec\n",
               perf->audio_duration_sec);
        printf("    Audio frames         %8d   (compression ratio ~%dx)\n",
               perf->audio_frames,
               perf->audio_frames > 0
                   ? (int)(n_samples / perf->audio_frames) : 0);

        printf("=========================================="
               "==========================================\n\n");
    }

    /* Cleanup */
    vv_free(live.ms);
    vv_free(live.enc_ms);
    vv_transcription_free(result);
    vv_inference_free(ctx);
    vv_free(audio);

    VV_LOG_I("Done.");
    return 0;
}
