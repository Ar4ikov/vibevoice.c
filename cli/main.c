/**
 * @file main.c
 * @brief CLI executable for vibevoice.c — VibeVoice-ASR inference.
 *
 * Usage:
 *   vv_cli --model <path> --audio <file.wav> [--output <file.json>]
 *          [--gpu <id>] [--max-tokens <N>] [--hotwords "word1,word2"]
 *          [--trt-acoustic <plan>] [--trt-semantic <plan>]
 *          [--kv-cache FMT] [--verbose]
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/kv_quant.h"
#include "vibevoice/audio.h"
#include "vibevoice/inference.h"

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
    const char* trt_acoustic;
    const char* trt_semantic;
    int         gpu_id;
    int         max_tokens;
    int         max_seq_len;
    int         gpu_layers;
    const char* kv_cache;
    bool        cpu_only;
    float       vram_budget;
    bool        verbose;
    const char* hotwords;
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
        "  --max-tokens <N>      Max decode tokens (default: 64000)\n"
        "  --max-seq-len <N>     KV-cache window in tokens (default: 32768)\n"
        "  --hotwords <words>    Comma-separated hotwords\n"
        "  --trt-acoustic <plan> TensorRT engine for acoustic encoder\n"
        "  --trt-semantic <plan> TensorRT engine for semantic encoder\n"
        "  --kv-cache FMT        KV-cache storage: fp16 (default), fp8,\n"
        "                        fp8-e5m2, tq4, tq3, tq2, tq1.5\n"
        "  --vram-budget <0-1>   VRAM fraction for model (default: 1.0)\n"
        "  --gpu-layers <N>      Layers to keep on the GPU (-1 = fit to VRAM)\n"
        "  --cpu                 CPU-only mode (no GPU)\n"
        "  --verbose             Enable debug logging\n"
        "  --help                Show this message\n\n"
        "Commands:\n"
        "  serve                 Run the OpenAI-compatible HTTP server\n"
        "  chat                  Interactive prompt, model stays loaded\n"
        "  mic                   Live microphone transcription\n"
        "  (none)                Transcribe one file and exit\n\n"
        "Run `%s <command> --help` for that command's own options.\n",
        VV_VERSION_STRING, prog, prog);
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
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            args->max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-seq-len") == 0 && i + 1 < argc) {
            args->max_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hotwords") == 0 && i + 1 < argc) {
            args->hotwords = argv[++i];
        } else if (strcmp(argv[i], "--trt-acoustic") == 0 && i + 1 < argc) {
            args->trt_acoustic = argv[++i];
        } else if (strcmp(argv[i], "--trt-semantic") == 0 && i + 1 < argc) {
            args->trt_semantic = argv[++i];
        } else if (strcmp(argv[i], "--kv-cache") == 0 && i + 1 < argc) {
            args->kv_cache = argv[++i];
        } else if (strcmp(argv[i], "--vram-budget") == 0 && i + 1 < argc) {
            args->vram_budget = (float)atof(argv[++i]);
            if (args->vram_budget < 0.0f) args->vram_budget = 0.0f;
            if (args->vram_budget > 1.0f) args->vram_budget = 1.0f;
        } else if (strcmp(argv[i], "--gpu-layers") == 0 && i + 1 < argc) {
            args->gpu_layers = atoi(argv[++i]);
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

int main(int argc, char** argv) {
#ifdef _WIN32
    /* Enable UTF-8 console output */
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

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

    VV_LOG_I("vibevoice.c v%s", VV_VERSION_STRING);
    VV_LOG_I("Model: %s", args.model_dir);
    VV_LOG_I("Audio: %s", args.audio_path);
    VV_LOG_I("GPU: %d", args.gpu_id);

    double t_start = get_time_ms();

    /* Step 1: Load and preprocess audio */
    double t0 = get_time_ms();
    float* audio = NULL;
    int n_samples = 0;

    vv_status_t s = vv_audio_preprocess(args.audio_path, &audio, &n_samples);
    if (s != VV_OK) {
        VV_LOG_E("Failed to load audio: %s", vv_status_str(s));
        return 1;
    }
    double t1 = get_time_ms();
    VV_LOG_I("Audio loaded: %d samples (%.2f sec) in %.1f ms",
             n_samples, (float)n_samples / 24000.0f, t1 - t0);

    /* Step 2: Initialize inference */
    t0 = get_time_ms();
    vv_inference_ctx_t* ctx = NULL;
    vv_init_params_t init_params = vv_init_params_default();
    if (args.kv_cache) {
        vv_kv_format_t f = vv_kv_format_parse(args.kv_cache);
        if (f >= VV_KV_FORMAT_COUNT) {
            fprintf(stderr, "error: unknown --kv-cache format '%s'\n",
                    args.kv_cache);
            return 1;
        }
        init_params.kv_format = (int)f;
    }
    init_params.vram_budget = args.vram_budget;
    init_params.cpu_only = args.cpu_only;
    if (args.max_seq_len > 0) init_params.max_seq_len = args.max_seq_len;
    init_params.gpu_layers = args.gpu_layers;
    s = vv_inference_init(args.model_dir, args.gpu_id, &init_params, &ctx);
    if (s != VV_OK) {
        VV_LOG_E("Failed to initialize inference: %s", vv_status_str(s));
        vv_free(audio);
        return 1;
    }
    t1 = get_time_ms();
    VV_LOG_I("Model loaded in %.1f ms", t1 - t0);

    /* Step 3: Transcribe */
    t0 = get_time_ms();
    vv_inference_params_t params;
    memset(&params, 0, sizeof(params));
    params.max_new_tokens = args.max_tokens;
    params.temperature = 0.0f;
    params.top_k = 1;
    params.enable_timestamps = true;
    params.enable_diarize = true;

    char hotword_buf[512];
    const char* hotword_list[32];
    params.num_hotwords = split_hotwords(args.hotwords, hotword_buf,
                                          sizeof(hotword_buf),
                                          hotword_list, 32);
    params.hotwords = params.num_hotwords ? hotword_list : NULL;
    if (params.num_hotwords)
        VV_LOG_I("Hotwords: %d term(s)", params.num_hotwords);

    vv_transcription_t* result = NULL;
    s = vv_inference_transcribe(ctx, audio, n_samples, &params, &result);
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
    vv_transcription_free(result);
    vv_inference_free(ctx);
    vv_free(audio);

    VV_LOG_I("Done.");
    return 0;
}
