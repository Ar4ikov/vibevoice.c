/**
 * @file cmd_chat.c
 * @brief `vv_cli chat` and `vv_cli mic` — the model stays loaded between clips.
 *
 * Both commands exist for the same reason: loading the 7B takes ~10 s, and
 * paying that per file makes short clips absurd. The engine comes up once and
 * then answers as fast as the audio allows.
 *
 * chat  a prompt loop: give it a path, or `rec 5` to record five seconds
 * mic   continuous capture, segmented by the VAD, transcribed as you speak
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/engine.h"
#include "vibevoice/capture.h"
#include "vibevoice/audio.h"
#include "vibevoice/kv_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

static volatile int g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

typedef struct {
    vv_engine_params_t ep;
    const char* device;
    const char* mic_file;      /* feed a WAV instead of a device */
    const char* hotwords;
    float start_db, stop_db, hangover_ms, max_segment_s;
    bool  timestamps;
    bool  verbose;
} chat_args_t;

static void usage(const char* which) {
    if (strcmp(which, "mic") == 0) {
        fprintf(stderr,
            "Usage: vv_cli mic --model <dir> [options]\n\n"
            "  --device <name>       Capture device (ALSA/dshow/avfoundation)\n"
            "  --from-file <wav>     Replay a WAV in real time instead of a mic\n"
            "  --start-db <dB>       Speech threshold (default: -38)\n"
            "  --stop-db <dB>        Silence threshold (default: -45)\n"
            "  --hangover <ms>       Silence before a segment closes (700)\n"
            "  --max-segment <s>     Force a cut after this long (30)\n"
            "  --timestamps          Print segment start/end times\n");
    } else {
        fprintf(stderr,
            "Usage: vv_cli chat --model <dir> [options]\n\n"
            "Commands inside the prompt:\n"
            "  <path>                Transcribe an audio file\n"
            "  rec <seconds>         Record from the microphone, then transcribe\n"
            "  hotwords a,b,c        Set hotwords for later requests\n"
            "  stats                 Engine counters\n"
            "  quit                  Exit\n");
    }
    fprintf(stderr,
        "\nCommon:\n"
        "  --model <dir>         Model directory (required)\n"
        "  --gpu <id>            GPU device id\n"
        "  --slots <n>           Concurrent transcriptions\n"
        "  --max-seq-len <n>     KV window\n"
        "  --kv-cache <fmt>      fp16 | fp8 | tq4 | ...\n"
        "  --hotwords a,b,c      Hotwords\n"
        "  --cpu                 CPU-only\n"
        "  --verbose\n");
}

static int parse_common(int argc, char** argv, chat_args_t* a,
                        const char* which) {
    a->ep = vv_engine_params_default();
    a->start_db = -38.0f;
    a->stop_db = -45.0f;
    a->hangover_ms = 700.0f;
    a->max_segment_s = 30.0f;

    for (int i = 0; i < argc; i++) {
        const char* s = argv[i];
        const char* next = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (strcmp(s, "--model") == 0 && next) a->ep.model_dir = argv[++i];
        else if (strcmp(s, "--gpu") == 0 && next) a->ep.gpu_id = atoi(argv[++i]);
        else if (strcmp(s, "--slots") == 0 && next) a->ep.n_slots = atoi(argv[++i]);
        else if (strcmp(s, "--max-seq-len") == 0 && next) a->ep.max_seq_len = atoi(argv[++i]);
        else if (strcmp(s, "--gpu-layers") == 0 && next) a->ep.gpu_layers = atoi(argv[++i]);
        else if (strcmp(s, "--vram-budget") == 0 && next) a->ep.vram_budget = (float)atof(argv[++i]);
        else if (strcmp(s, "--kv-cache") == 0 && next) {
            vv_kv_format_t f = vv_kv_format_parse(argv[++i]);
            if (f >= VV_KV_FORMAT_COUNT) {
                fprintf(stderr, "error: unknown --kv-cache format '%s'\n", argv[i]);
                return -1;
            }
            a->ep.kv_format = (int)f;
        }
        else if (strcmp(s, "--device") == 0 && next) a->device = argv[++i];
        else if (strcmp(s, "--from-file") == 0 && next) a->mic_file = argv[++i];
        else if (strcmp(s, "--hotwords") == 0 && next) a->hotwords = argv[++i];
        else if (strcmp(s, "--start-db") == 0 && next) a->start_db = (float)atof(argv[++i]);
        else if (strcmp(s, "--stop-db") == 0 && next) a->stop_db = (float)atof(argv[++i]);
        else if (strcmp(s, "--hangover") == 0 && next) a->hangover_ms = (float)atof(argv[++i]);
        else if (strcmp(s, "--max-segment") == 0 && next) a->max_segment_s = (float)atof(argv[++i]);
        else if (strcmp(s, "--timestamps") == 0) a->timestamps = true;
        else if (strcmp(s, "--cpu") == 0) a->ep.cpu_only = true;
        else if (strcmp(s, "--verbose") == 0) a->verbose = true;
        else if (strcmp(s, "--help") == 0 || strcmp(s, "-h") == 0) { usage(which); return 1; }
        else {
            fprintf(stderr, "%s: unknown option '%s'\n\n", which, s);
            usage(which);
            return -1;
        }
    }
    if (!a->ep.model_dir) {
        fprintf(stderr, "%s: --model is required\n\n", which);
        usage(which);
        return -1;
    }
    return 0;
}

/** @brief Fill inference params, pointing hotwords into `scratch`. */
static void build_params(vv_inference_params_t* p, const char* csv,
                         char* scratch, size_t scratch_n,
                         const char** list, int max_list) {
    memset(p, 0, sizeof(*p));
    p->max_new_tokens = 64000;
    p->temperature = 0.0f;
    p->top_k = 1;
    p->enable_timestamps = true;
    p->enable_diarize = true;
    p->num_hotwords = 0;

    if (!csv || !csv[0]) return;
    snprintf(scratch, scratch_n, "%s", csv);
    int n = 0;
    char* tok = strtok(scratch, ",");
    while (tok && n < max_list) {
        while (*tok == ' ') tok++;
        if (*tok) list[n++] = tok;
        tok = strtok(NULL, ",");
    }
    p->hotwords = n ? list : NULL;
    p->num_hotwords = n;
}

static void print_result(const vv_transcription_t* tr,
                         const vv_perf_metrics_t* perf, bool timestamps) {
    if (!tr) return;
    /*
     * Fall back to the flat text when the model's answer did not parse into
     * segments -- it still holds the transcription, and dropping it because
     * the JSON was malformed would be the wrong trade.
     */
    if (timestamps && tr->num_segments > 0) {
        for (int i = 0; i < tr->num_segments; i++) {
            const vv_segment_t* s = &tr->segments[i];
            printf("[%6.2f - %6.2f] %-10s %s\n", s->start_time, s->end_time,
                   s->speaker ? s->speaker : "", s->text ? s->text : "");
        }
    } else {
        printf("%s\n", tr->full_text ? tr->full_text : "");
    }
    if (perf)
        fprintf(stderr, "  (%.2fs audio, %.0f ms, RTF %.3f, %.0f tok/s)\n",
                tr->duration, perf->total_ms, perf->rtf,
                perf->decode_tok_per_sec);
    fflush(stdout);
}

/* ─── mic ────────────────────────────────────────────────────────────────── */

int vv_cmd_mic(int argc, char** argv) {
    chat_args_t a;
    memset(&a, 0, sizeof(a));
    const int pr = parse_common(argc, argv, &a, "mic");
    if (pr != 0) return pr > 0 ? 0 : 1;

    vv_log_set_level(a.verbose ? VV_LOG_DEBUG : VV_LOG_WARN);

    vv_engine_t* engine = NULL;
    vv_status_t s = vv_engine_create(&a.ep, &engine);
    if (s != VV_OK) {
        fprintf(stderr, "mic: cannot start engine: %s\n", vv_status_str(s));
        return 1;
    }

    vv_mic_t* mic = NULL;
    if (a.mic_file)
        s = vv_mic_open_file(a.mic_file, 24000, true, &mic);
    else
        s = vv_mic_open(24000, a.device, &mic);
    if (s != VV_OK) {
        fprintf(stderr, "mic: cannot open capture: %s\n", vv_status_str(s));
        vv_engine_free(engine);
        return 1;
    }

    vv_vad_params_t vp = vv_vad_params_default();
    vp.sample_rate = 24000;
    vp.start_db = a.start_db;
    vp.stop_db = a.stop_db;
    vp.hangover_ms = a.hangover_ms;
    vp.max_segment_s = a.max_segment_s;

    vv_vad_t* vad = NULL;
    if (vv_vad_create(&vp, &vad) != VV_OK) {
        vv_mic_close(mic);
        vv_engine_free(engine);
        return 1;
    }

    signal(SIGINT, on_signal);
    fprintf(stderr, "listening via %s (Ctrl-C to stop)\n", vv_mic_backend(mic));

    char hot_scratch[512];
    const char* hot_list[32];
    vv_inference_params_t ip;
    build_params(&ip, a.hotwords, hot_scratch, sizeof(hot_scratch),
                 hot_list, 32);

    float buf[4096];
    int n_segments = 0;
    while (!g_stop) {
        const int got = vv_mic_read(mic, buf, (int)(sizeof(buf) / sizeof(buf[0])));
        if (got == 0) {
            if (vv_mic_eof(mic)) break;
            vv_msleep(5);
            continue;
        }

        float* seg = NULL;
        int seg_len = 0;
        bool have = vv_vad_push(vad, buf, got, &seg, &seg_len);
        while (have) {
            vv_transcription_t* tr = NULL;
            vv_perf_metrics_t perf;
            memset(&perf, 0, sizeof(perf));
            if (vv_engine_transcribe(engine, seg, seg_len, 24000, &ip,
                                     &tr, &perf) == VV_OK) {
                n_segments++;
                print_result(tr, &perf, a.timestamps);
                vv_transcription_free(tr);
            }
            vv_free(seg);
            seg = NULL;
            have = vv_vad_drain(vad, &seg, &seg_len);
        }
    }

    /* Whatever is still open at the end is still worth transcribing. */
    {
        float* seg = NULL;
        int seg_len = 0;
        if (vv_vad_flush(vad, &seg, &seg_len)) {
            vv_transcription_t* tr = NULL;
            vv_perf_metrics_t perf;
            memset(&perf, 0, sizeof(perf));
            if (vv_engine_transcribe(engine, seg, seg_len, 24000, &ip,
                                     &tr, &perf) == VV_OK) {
                n_segments++;
                print_result(tr, &perf, a.timestamps);
                vv_transcription_free(tr);
            }
            vv_free(seg);
        }
    }

    const uint64_t over = vv_mic_overruns(mic);
    if (over)
        fprintf(stderr, "mic: dropped %llu samples (capture outran decode)\n",
                (unsigned long long)over);
    fprintf(stderr, "mic: %d utterance(s)\n", n_segments);

    vv_vad_free(vad);
    vv_mic_close(mic);
    vv_engine_free(engine);
    return 0;
}

/* ─── chat ───────────────────────────────────────────────────────────────── */

static void transcribe_path(vv_engine_t* engine, const char* path,
                            const vv_inference_params_t* ip, bool timestamps) {
    float* pcm = NULL;
    int n = 0, sr = 0;
    vv_status_t s = vv_audio_load_any(path, &pcm, &n, &sr);
    if (s != VV_OK) {
        printf("cannot read '%s': %s\n", path, vv_status_str(s));
        return;
    }
    vv_transcription_t* tr = NULL;
    vv_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));
    s = vv_engine_transcribe(engine, pcm, n, sr, ip, &tr, &perf);
    vv_free(pcm);
    if (s != VV_OK) {
        printf("transcription failed: %s\n", vv_status_str(s));
        return;
    }
    print_result(tr, &perf, timestamps);
    vv_transcription_free(tr);
}

static void record_and_transcribe(vv_engine_t* engine, const char* device,
                                  float seconds,
                                  const vv_inference_params_t* ip,
                                  bool timestamps) {
    vv_mic_t* mic = NULL;
    if (vv_mic_open(24000, device, &mic) != VV_OK) {
        printf("no microphone available\n");
        return;
    }
    const int want = (int)(seconds * 24000.0f);
    float* pcm = (float*)vv_alloc((size_t)want * sizeof(float));
    if (!pcm) { vv_mic_close(mic); return; }

    printf("recording %.1f s...\n", seconds);
    fflush(stdout);
    int have = 0;
    while (have < want) {
        const int got = vv_mic_read(mic, pcm + have, want - have);
        if (got == 0) vv_msleep(5);
        have += got;
    }
    vv_mic_close(mic);

    vv_transcription_t* tr = NULL;
    vv_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));
    if (vv_engine_transcribe(engine, pcm, have, 24000, ip, &tr, &perf) == VV_OK) {
        print_result(tr, &perf, timestamps);
        vv_transcription_free(tr);
    }
    vv_free(pcm);
}

int vv_cmd_chat(int argc, char** argv) {
    chat_args_t a;
    memset(&a, 0, sizeof(a));
    const int pr = parse_common(argc, argv, &a, "chat");
    if (pr != 0) return pr > 0 ? 0 : 1;

    vv_log_set_level(a.verbose ? VV_LOG_DEBUG : VV_LOG_WARN);

    vv_engine_t* engine = NULL;
    vv_status_t s = vv_engine_create(&a.ep, &engine);
    if (s != VV_OK) {
        fprintf(stderr, "chat: cannot start engine: %s\n", vv_status_str(s));
        return 1;
    }

    char hot_scratch[512];
    const char* hot_list[32];
    char hot_csv[512];
    snprintf(hot_csv, sizeof(hot_csv), "%s", a.hotwords ? a.hotwords : "");
    vv_inference_params_t ip;
    build_params(&ip, hot_csv, hot_scratch, sizeof(hot_scratch), hot_list, 32);

    printf("vibevoice.c %s — model ready, %d slot(s).\n"
           "Type a file path, `rec <seconds>`, `hotwords a,b`, `stats`, "
           "or `quit`.\n",
           VV_VERSION_STRING, vv_engine_slots(engine));

    char line[1024];
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;

        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        while (n && line[n - 1] == ' ') line[--n] = '\0';
        char* cmd = line;
        while (*cmd == ' ') cmd++;
        if (!*cmd) continue;

        if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "exit") == 0 ||
            strcmp(cmd, "/quit") == 0) break;

        if (strcmp(cmd, "stats") == 0) {
            uint64_t done = 0;
            int busy = 0;
            vv_engine_stats(engine, &done, &busy);
            printf("slots %d, busy %d, completed %llu\n",
                   vv_engine_slots(engine), busy, (unsigned long long)done);
            continue;
        }
        if (strncmp(cmd, "hotwords", 8) == 0) {
            const char* rest = cmd + 8;
            while (*rest == ' ') rest++;
            snprintf(hot_csv, sizeof(hot_csv), "%s", rest);
            build_params(&ip, hot_csv, hot_scratch, sizeof(hot_scratch),
                         hot_list, 32);
            printf("hotwords: %d term(s)\n", ip.num_hotwords);
            continue;
        }
        if (strncmp(cmd, "rec", 3) == 0 && (cmd[3] == ' ' || cmd[3] == '\0')) {
            const float secs = (cmd[3] == ' ') ? (float)atof(cmd + 4) : 5.0f;
            record_and_transcribe(engine, a.device,
                                  secs > 0.5f ? secs : 5.0f, &ip, a.timestamps);
            continue;
        }

        transcribe_path(engine, cmd, &ip, a.timestamps);
    }

    vv_engine_free(engine);
    return 0;
}
